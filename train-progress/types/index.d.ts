export type JobStatus = 'running' | 'done' | 'failed' | 'stalled'

export type Job = {
  id: string
  name: string
  status: JobStatus
  step: number
  total: number
  metric?: number
  metricName?: string
  etaSec?: number
  updatedMs: number
  host?: string
  session?: string
}

export type Scope = 'all' | 'session'

declare module 'claude-code' {
  interface PluginState {
    'train-progress': { jobs: Job[]; hideDone: boolean; dir: string; scope: Scope; sessionId: string }
  }
}
