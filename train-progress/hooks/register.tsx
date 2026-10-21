import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { Job, JobStatus, Scope } from '../types'

const PANE = 'train-progress'
const TITLE = 'Progress'
const POLL_MS = 5000
// How often to try starting the TensorBoard watcher; it exits at once when
// another session (or tp_view.py) already runs one on this machine.
const WATCHER_RETRY_MS = 30_000
// A running job whose file has not changed for this long is shown as stalled.
const STALL_MS = 15 * 60 * 1000
// A finished, failed or stalled job drops off the panel once it is this old.
const KEEP_MS = 24 * 60 * 60 * 1000
// ...and off the bar under the prompt once it is this old.
const BAR_KEEP_MS = 10 * 60 * 1000
// Rows of jobs the bar under the prompt shows before "n more".
const BAR_ROWS = 3
// No ETA until a job has run this long: `tp set` alone would give "ETA 0s".
const ETA_MIN_S = 5

const jobs = atom({ plugin: 'train-progress', key: 'jobs' } as const, [] as Job[])
const hideDone = atom({ plugin: 'train-progress', key: 'hideDone' } as const, false)
const isBarHidden = atom({ plugin: 'train-progress', key: 'isBarHidden' } as const, false)
const dir = atom({ plugin: 'train-progress', key: 'dir' } as const, '')
const scope = atom({ plugin: 'train-progress', key: 'scope' } as const, 'all' as Scope)
const sessionId = atom({ plugin: 'train-progress', key: 'sessionId' } as const, '')

const ICON: Record<JobStatus, string> = { running: '●', done: '✓', failed: '✗', stalled: '◌' }
const COLOR: Record<JobStatus, string> = {
  running: 'suggestion',
  done: 'success',
  failed: 'error',
  stalled: 'warning',
}

function duration(sec: number): string {
  const s = Math.max(0, Math.round(sec))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m${String(s % 60).padStart(2, '0')}s`
  const h = Math.floor(m / 60)
  if (h < 48) return `${h}h${String(m % 60).padStart(2, '0')}m`
  return `${Math.floor(h / 24)}d${h % 24}h`
}

function bar(fraction: number, width: number): string {
  const cells = Math.max(4, width)
  const filled = Math.round(Math.min(1, Math.max(0, fraction)) * cells)
  return '█'.repeat(filled) + '░'.repeat(cells - filled)
}

function parseJob(id: string, text: string, mtimeMs: number, now: number): Job | undefined {
  let raw: Record<string, unknown>
  try {
    raw = JSON.parse(text)
  } catch {
    return undefined
  }
  const step = Number(raw.step ?? 0)
  const total = Number(raw.total ?? 0)
  const started = Number(raw.started_at)
  const updated = Number(raw.updated_at ?? mtimeMs / 1000)
  // When the job last made progress: tb_watch.py rewrites its files on every
  // scan, so the file's own time says nothing about a hung run.
  const lastMs = updated > 0 ? updated * 1000 : mtimeMs
  let status: JobStatus =
    raw.status === 'done' || raw.status === 'failed' ? raw.status : 'running'
  if (status === 'running' && now - lastMs > STALL_MS) status = 'stalled'
  if (status !== 'running' && now - lastMs > KEEP_MS) return undefined

  let etaSec: number | undefined
  if (status === 'running' && step > 0 && total > step && started > 0 && updated - started >= ETA_MIN_S) {
    etaSec = ((total - step) * (updated - started)) / step
  }

  return {
    id,
    name: typeof raw.name === 'string' ? raw.name : id,
    status,
    step,
    total,
    // `metric` + `metric_name`, or the `loss` of helpers before 0.3
    metric: typeof raw.metric === 'number' ? raw.metric : typeof raw.loss === 'number' ? raw.loss : undefined,
    metricName: typeof raw.metric_name === 'string' ? raw.metric_name : 'loss',
    note: typeof raw.note === 'string' && raw.note !== '' ? raw.note : undefined,
    etaSec,
    startedMs: started > 0 ? started * 1000 : undefined,
    updatedMs: lastMs,
    host: typeof raw.host === 'string' ? raw.host : undefined,
    session: typeof raw.session === 'string' ? raw.session : undefined,
  }
}

function inScope(list: Job[], view: Scope, id: string): Job[] {
  return view === 'session' ? list.filter(j => j.session === id) : list
}

function fractionOf(job: Job): number | undefined {
  if (job.status === 'done') return 1
  return job.total > 0 ? job.step / job.total : undefined
}

// ETA while running, elapsed time when there is no total, else how long ago it ended.
function timing(job: Job, now: number): string {
  if (job.status !== 'running') return `${job.status} ${duration((now - job.updatedMs) / 1000)} ago`
  if (job.etaSec !== undefined) return `ETA ${duration(job.etaSec)}`
  if (job.total === 0 && job.startedMs !== undefined) return duration((now - job.startedMs) / 1000)
  return ''
}

function details(job: Job): string {
  const count = job.total > 0 ? `${job.step}/${job.total}` : job.step > 0 ? `step ${job.step}` : ''
  return [
    count,
    job.metric !== undefined && `${job.metricName} ${job.metric.toPrecision(4)}`,
    // `tp run` notes are often just the step line ("16/30")
    job.note !== count && job.note,
  ]
    .filter(Boolean)
    .join(' · ')
}

export const register: Register = (on, options) => {
  // The module's own memory of each job's last status, to toast changes.
  // A reload starts it over, which only means no toast on the first poll.
  const seen = new Map<string, JobStatus>()
  let isWatching = false

  on('session.start', async ($, e, next) => {
    const home = (await $.env.get('HOME')) ?? ''
    const custom = await $.env.get('TRAIN_PROGRESS_DIR')
    const folder = custom ?? `${home}/.train_progress`
    const id = await $.session.id()
    await update($, dir, () => folder)
    await update($, sessionId, () => id)
    await update($, scope, () => (options.scope === 'session' ? 'session' : 'all'))

    for (const name of ['progress', 'train']) {
      // A name another command holds is refused; the bar still works without it.
      await $.command
        .register({
          name,
          description: 'Progress of training runs and other long jobs (all|session, panel|close, hide|show)',
          argumentHint: '[all|session|panel|close|hide|show]',
        })
        .catch(() => undefined)
    }

    if (options.watcher !== false) {
      // Tell the watcher where this project's TensorBoard logs are.
      const wanted = String(options.logRoots ?? '')
        .split(',')
        .map(s => s.trim())
        .filter(Boolean)
      if (await $.fs.exists(`${e.cwd}/logs`)) wanted.push(`${e.cwd}/logs`)
      const rootsFile = `${folder}/roots.txt`
      const known = (await $.fs.read(rootsFile).catch(() => '')) as string
      const lines = known.split('\n').filter(Boolean)
      const added = wanted.filter(root => !lines.includes(root))
      if (added.length > 0) await $.fs.write(rootsFile, [...lines, ...added].join('\n') + '\n')

      const watcher = `${$.plugin.root}/python/tb_watch.py`
      const startWatcher = () => {
        if (isWatching) return
        isWatching = true
        void (async () => {
          try {
            for await (const _ of $.process.spawn({ argv: ['python3', watcher] })) {
              // the watcher writes files, not output
            }
          } catch {
            // no python3: helper-reported jobs still show
          }
          isWatching = false
        })()
      }
      startWatcher()
      $.clock.every(WATCHER_RETRY_MS, startWatcher)
    }

    const poll = async () => {
      const now = Date.now()
      const entries = await $.fs.list(folder).catch(() => [])
      const found: Job[] = []
      for (const entry of entries) {
        if (entry.kind !== 'file' || !entry.name.endsWith('.json')) continue
        const text = await $.fs.read(`${folder}/${entry.name}`).catch(() => undefined)
        if (typeof text !== 'string') continue
        const job = parseJob(entry.name.slice(0, -5), text, entry.mtimeMs, now)
        if (job) found.push(job)
      }
      found.sort((a, b) => b.updatedMs - a.updatedMs)

      const shown = inScope(found, await read($, scope), id)
      for (const job of shown) {
        const before = seen.get(job.id)
        if (before !== undefined && before !== job.status && job.status !== 'running') {
          void $.ui.toast(`${ICON[job.status]} ${job.name} ${job.status}`)
        }
        seen.set(job.id, job.status)
      }

      // The bar under the prompt says all this; the status line stands in when it is hidden.
      const running = shown.filter(j => j.status === 'running').length
      const failed = shown.filter(j => j.status === 'failed' || j.status === 'stalled').length
      const parts = [running > 0 && `${running} running`, failed > 0 && `${failed} need attention`]
      const summary = parts.filter(Boolean).join(' · ')
      void $.ui.status(summary === '' || !(await read($, isBarHidden)) ? undefined : summary)

      await update($, jobs, () => found)
    }

    await poll()
    $.clock.every(POLL_MS, () => void poll())

    return next(e)
  })

  on('command.run', { command: ['progress', 'train'] }, async ($, e) => {
    const arg = e.args.trim()
    if (arg === 'all' || arg === 'session') await update($, scope, () => arg)
    if (arg === 'panel') {
      await $.ui.open({ id: PANE, title: TITLE })
      return { text: 'Progress panel opened. /progress close closes it.' }
    }
    if (arg === 'close') {
      await $.ui.close({ id: PANE })
      return { text: 'Progress panel closed.' }
    }
    await update($, isBarHidden, () => arg === 'hide')
    if (arg === 'hide') return { text: 'Progress bar hidden; the status line keeps a count. /progress shows it again.' }
    const view = await read($, scope)

    return { text: `Showing ${view === 'all' ? 'every job on this machine' : 'this session\'s jobs'} under the prompt.` }
  })

  // The rows go under the prompt, beneath the engine's own hint line, which stays live.
  on('ui.render', { component: 'PromptHint' }, async ($, e, next) => {
    const hint = await next(e)
    if (await read($, isBarHidden)) return hint
    const now = Date.now()
    // Running jobs first, then the ones that ended in the last few minutes.
    const recent = inScope(await read($, jobs), await read($, scope), await read($, sessionId)).filter(
      j => j.status === 'running' || now - j.updatedMs < BAR_KEEP_MS,
    )
    const list = [...recent.filter(j => j.status === 'running'), ...recent.filter(j => j.status !== 'running')]
    if (list.length === 0) return hint

    const { Box, Text } = $.ui.resolve(e)
    const width = (e.viewport?.columns ?? 80) - 4
    // name · bar · percent · details, the details cut first as the row narrows
    const nameCols = Math.min(28, Math.max(10, Math.floor(width * 0.3)))
    const barCols = width >= 60 ? 16 : 8
    const rows = list.slice(0, BAR_ROWS)

    return (
      <Box flexDirection="column">
        {hint}
        {rows.map(job => {
          const fraction = fractionOf(job)
          const right = [details(job), timing(job, now)].filter(Boolean).join(' · ')
          return (
            <Box key={job.id} flexDirection="row" gap={1} paddingLeft={2}>
              <Box width={nameCols}>
                <Text color={COLOR[job.status]} wrap="truncate-end">
                  {ICON[job.status]} {job.name}
                </Text>
              </Box>
              <Text color={COLOR[job.status]}>{fraction === undefined ? '░'.repeat(barCols) : bar(fraction, barCols)}</Text>
              <Text tabular>{fraction === undefined ? '   …' : `${Math.floor(fraction * 100)}%`.padStart(4)}</Text>
              <Text dimColor wrap="truncate-end">
                {right}
              </Text>
            </Box>
          )
        })}
        {list.length > rows.length && (
          <Box paddingLeft={2}>
            <Text dimColor>+{list.length - rows.length} more · /progress panel</Text>
          </Box>
        )}
      </Box>
    )
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button } = $.ui.resolve(e)
    const view = await read($, scope)
    const list = inScope(await read($, jobs), view, await read($, sessionId))
    const isHiding = await read($, hideDone)
    const folder = await read($, dir)
    const width = e.props.bodyColumns ?? 40
    const shown = isHiding ? list.filter(j => j.status !== 'done') : list
    const now = Date.now()

    const controls = (
      <Box flexDirection="row" gap={2} marginBottom={1}>
        <Button
          key="scope"
          plain
          onPress={() => update($, scope, v => (v === 'all' ? 'session' : 'all'))}
        >
          {view === 'all' ? 'all sessions' : 'this session'} <Text dimColor>⇄</Text>
        </Button>
        <Button key="toggle-done" plain onPress={() => update($, hideDone, v => !v)}>
          <Text dimColor>{isHiding ? 'show finished' : 'hide finished'}</Text>
        </Button>
      </Box>
    )

    if (list.length === 0) {
      return (
        <Box flexDirection="column">
          {controls}
          <Text dimColor>
            {view === 'session' ? 'No jobs started from this session.' : 'No jobs found in'}
          </Text>
          {view === 'all' && <Text dimColor>{folder}</Text>}
          <Text dimColor> </Text>
          <Text dimColor>TensorBoard runs under ./logs show up by themselves;</Text>
          <Text dimColor>`tp run -- CMD` tracks any command.</Text>
        </Box>
      )
    }

    return (
      <Box flexDirection="column">
        {controls}
        {shown.map(job => {
          const fraction = fractionOf(job)
          const pct = fraction === undefined ? '   …' : `${Math.floor(fraction * 100)}%`.padStart(4)
          const line = [details(job), job.host].filter(Boolean).join(' · ')

          return (
            <Box key={job.id} flexDirection="column" marginBottom={1}>
              <Box flexDirection="row" justifyContent="space-between">
                <Text color={COLOR[job.status]} wrap="truncate-end">
                  {ICON[job.status]} <Text weight="bold">{job.name}</Text>
                </Text>
                <Text dimColor>{timing(job, now)}</Text>
              </Box>
              <Text>
                <Text color={COLOR[job.status]}>{bar(fraction ?? 0, width - 6)}</Text>
                <Text tabular>{pct}</Text>
              </Text>
              <Text dimColor wrap="truncate-end">
                {line}
              </Text>
            </Box>
          )
        })}
        {shown.length === 0 && <Text dimColor>All jobs finished.</Text>}
      </Box>
    )
  })
}
