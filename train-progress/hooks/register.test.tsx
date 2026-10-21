import { expect, test } from 'claude-code/testing'

const PANE = {
  plugin: 'train-progress',
  component: 'Pane',
  requestId: 'train-progress',
  props: {
    title: 'Training',
    isFocused: false,
    bodyColumns: 40,
    placement: 'dock',
    scroll: { offset: 0, bodyRows: 20 },
    view: {},
  },
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
