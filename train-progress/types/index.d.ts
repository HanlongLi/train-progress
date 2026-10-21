export type JobStatus = 'running' | 'done' | 'failed' | 'stalled'

export type Job = {
  id: string
  name: string
  status: JobStatus
  step: number
  total: number
  metric?: number
  metricName?: string
  note?: string
  etaSec?: number
  startedMs?: number
  updatedMs: number
  host?: string
  session?: string
}

export type Scope = 'all' | 'session'

declare module 'claude-code' {
  interface PluginState {
    'train-progress': {
      jobs: Job[]
      hideDone: boolean
      isBarHidden: boolean
      dir: string
      scope: Scope
      sessionId: string
    }
  }
}
