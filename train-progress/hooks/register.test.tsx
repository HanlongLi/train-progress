import { expect, test } from 'claude-code/testing'

const PANE = {
  plugin: 'train-progress',
  component: 'Pane',
  requestId: 'train-progress',
  props: {
    title: 'Progress',
    isFocused: false,
    bodyColumns: 40,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 20 },
    view: {},
  },
} as const

const HINT = {
  plugin: 'train-progress',
  component: 'PromptHint',
  viewport: { columns: 110, rows: 40 },
  props: { isDraft: false, isWorking: false, hint: '? for shortcuts' },
} as const

const now = Date.now()

function files(session: string): Record<string, string> {
  return {
    'bert.json': JSON.stringify({
      name: 'bert-ft', step: 600, total: 1000, loss: 0.4213, status: 'running',
      started_at: now / 1000 - 600, updated_at: now / 1000,
    }),
    'vit.json': JSON.stringify({ name: 'vit-l', step: 50, total: 50, status: 'done' }),
    'rl.json': JSON.stringify({
      name: 'bimanual', step: 562, total: 3000, metric: 42.13, metric_name: 'reward',
      status: 'running', session,
    }),
    'sync.json': JSON.stringify({
      name: 'rsync photos', step: 0, total: 0, note: 'IMG_2041.jpg', status: 'running',
      started_at: now / 1000 - 125, updated_at: now / 1000,
    }),
  }
}

test('the pane draws a bar per job, on every surface', { options: { watcher: false } }, async ($, on) => {
  const FILES = files('other-session')
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/home/me' : undefined }))
  on('fs.list', () => ({
    value: Object.keys(FILES).map(name => ({ name, kind: 'file', size: 1, mtimeMs: now, isLink: false })),
  }))
  on('fs.read', (_, e) => ({ value: FILES[e.path.split('/').pop() ?? ''] ?? '' }))
  on('command.register', () => ({ value: { command: 'train' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.id', () => ({ value: 'this-session' }))

  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })

  for (const surface of ['terminal', 'desktop', 'vscode', 'mobile'] as const) {
    const ui = await $.ui.mount({ ...PANE, surface })
    expect(await ui.find({ type: 'Text', text: /60%/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /ETA 6m40s/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /loss 0\.4213/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /vit-l/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /562\/3000 · reward 42\.13/ })).toBeDefined()
    await ui.unmount()
  }
})

test('the session view shows only this session\'s jobs', { options: { watcher: false, scope: 'session' } }, async ($, on) => {
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/home/me' : undefined }))
  on('command.register', () => ({ value: { command: 'train' } }))
  on('ui.open', () => ({ value: { isPlaced: true } }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.id', () => ({ value: 'this-session' }))
  const FILES = files('this-session')
  on('fs.list', () => ({
    value: Object.keys(FILES).map(name => ({ name, kind: 'file', size: 1, mtimeMs: now, isLink: false })),
  }))
  on('fs.read', (_, e) => ({ value: FILES[e.path.split('/').pop() ?? ''] ?? '' }))

  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })

  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({ ...PANE, surface })
    expect(await ui.find({ type: 'Text', text: /bimanual/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /bert-ft/ })).toBeUndefined()

    await ui.press({ key: 'scope' })
    expect(await ui.find({ type: 'Text', text: /bert-ft/ })).toBeDefined()
    await ui.press({ key: 'scope' })
    await ui.unmount()
  }
})

test('/progress panel opens the panel and /progress close closes it', { options: { watcher: false } }, async ($, on) => {
  const calls: string[] = []
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/home/me' : undefined }))
  on('fs.list', () => ({ value: [] }))
  on('command.register', () => ({ value: { command: 'progress' } }))
  on('ui.open', (_, e) => (calls.push(`open ${e.id}`), { value: { isPlaced: true } }))
  on('ui.close', (_, e) => (calls.push(`close ${e.id}`), { value: undefined }))
  on('ui.status', () => ({ value: undefined }))
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.id', () => ({ value: 'this-session' }))

  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })
  const run = (args: string) =>
    $.command.run({
      command: 'progress',
      args,
      origin: { kind: 'composer' },
      presentation: { isFullscreen: false, columns: 110 },
    })

  expect((await run('panel')).text).toMatch(/opened/)
  expect((await run('close')).text).toMatch(/closed/)
  expect(calls).toEqual(['open train-progress', 'close train-progress'])
})

test('the bar under the prompt keeps the hint line, shows a row per job, and hides', { options: { watcher: false } }, async ($, on) => {
  const FILES = files('other-session')
  on('env.get', (_, e) => ({ value: e.name === 'HOME' ? '/home/me' : undefined }))
  on('fs.list', () => ({
    value: Object.keys(FILES).map(name => ({ name, kind: 'file', size: 1, mtimeMs: now, isLink: false })),
  }))
  on('fs.read', (_, e) => ({ value: FILES[e.path.split('/').pop() ?? ''] ?? '' }))
  on('command.register', () => ({ value: { command: 'progress' } }))
  on('ui.status', () => ({ value: undefined }))
  // stands for the engine's own hint line
  on('ui.render', ($, e) => {
    const { Text } = $.ui.resolve(e)
    return <Text dimColor>? for shortcuts</Text>
  })
  on('session.start', (_, e) => ({ cwd: e.cwd }))
  on('session.id', () => ({ value: 'this-session' }))

  await $.session.start({ cwd: '/', surface: 'terminal', isInteractive: true })

  for (const surface of ['terminal', 'desktop'] as const) {
    const ui = await $.ui.mount({ ...HINT, surface })
    expect(await ui.find({ type: 'Text', text: /\? for shortcuts/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /bert-ft/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /60%/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /600\/1000 · loss 0\.4213 · ETA 6m40s/ })).toBeDefined()
    // no total: elapsed time and the last note, no percent
    expect(await ui.find({ type: 'Text', text: /IMG_2041\.jpg · 2m05s/ })).toBeDefined()
    expect(await ui.find({ type: 'Text', text: /\+1 more/ })).toBeDefined()
    await ui.unmount()
  }

  await $.command.run({
    command: 'progress',
    args: 'hide',
    origin: { kind: 'composer' },
    presentation: { isFullscreen: false, columns: 110 },
  })
  const hidden = await $.ui.mount({ ...HINT, surface: 'terminal' })
  expect(await hidden.find({ type: 'Text', text: /bert-ft/ })).toBeUndefined()
  expect(await hidden.find({ type: 'Text', text: /\? for shortcuts/ })).toBeDefined()
  await hidden.unmount()
})
