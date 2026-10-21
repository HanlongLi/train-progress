// train-progress for OpenCode: toasts when a training run or other tracked job
// finishes, fails or stalls, a `progress` tool the model can read jobs with, and
// the session id in every shell so `tp --session` and the "session" scope work.
// install.sh copies this file to ~/.config/opencode/plugins/.
//
// TRAIN_PROGRESS_SCOPE=session toasts only jobs started from this OpenCode process.

import { execFile, spawn } from "node:child_process"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

const POLL_MS = 5000
// How often to try starting the TensorBoard watcher; it exits at once when
// another one (Claude Code, Codex, tp) already runs on this machine.
const WATCHER_RETRY_MS = 30_000
const STALL_S = 15 * 60
const KEEP_S = 24 * 3600
const ICON = { running: "●", done: "✓", failed: "✗", stalled: "◌" }
const VARIANT = { done: "success", failed: "error", stalled: "warning" }

const jobsDir = () => process.env.TRAIN_PROGRESS_DIR || path.join(os.homedir(), ".train_progress")
// Where install.sh put the python side (tp_view.py, tb_watch.py).
const pythonDir = () =>
  process.env.TRAIN_PROGRESS_HOME ||
  path.join(process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"), "train-progress")

function load(dir) {
  const now = Date.now() / 1000
  const jobs = []
  let names = []
  try {
    names = fs.readdirSync(dir).filter(n => n.endsWith(".json"))
  } catch {
    return jobs
  }
  for (const name of names) {
    let raw, mtime
    try {
      mtime = fs.statSync(path.join(dir, name)).mtimeMs / 1000
      raw = JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"))
    } catch {
      continue
    }
    const last = Number(raw.updated_at) || mtime
    let status = raw.status === "done" || raw.status === "failed" ? raw.status : "running"
    if (status === "running" && now - last > STALL_S) status = "stalled"
    if (status !== "running" && now - last > KEEP_S) continue
    jobs.push({ ...raw, id: name.slice(0, -5), status })
  }
  return jobs
}

function summary(job) {
  const step = Number(job.step) || 0
  const total = Number(job.total) || 0
  const parts = [`${ICON[job.status]} ${job.name ?? job.id}`]
  if (job.status === "running" && total > 0) parts.push(`${Math.min(100, Math.floor((step * 100) / total))}%`)
  else if (job.status !== "running") parts.push(job.status)
  if (job.note) parts.push(`(${String(job.note).slice(0, 60)})`)
  return parts.join(" ")
}

export const TrainProgress = async ({ client, directory }) => {
  const dir = jobsDir()
  const sessions = new Set()
  const scoped = () => process.env.TRAIN_PROGRESS_SCOPE === "session"
  const inScope = jobs => (scoped() ? jobs.filter(j => sessions.has(j.session)) : jobs)

  // Tell the watcher where this project's TensorBoard logs are.
  const logs = path.join(directory, "logs")
  if (fs.existsSync(logs)) {
    const rootsFile = path.join(dir, "roots.txt")
    let known = []
    try {
      known = fs.readFileSync(rootsFile, "utf8").split("\n")
    } catch {}
    if (!known.includes(logs)) {
      fs.mkdirSync(dir, { recursive: true })
      fs.appendFileSync(rootsFile, logs + "\n")
    }
  }

  let isWatching = false
  const startWatcher = () => {
    if (isWatching) return
    isWatching = true
    try {
      const child = spawn("python3", [path.join(pythonDir(), "tb_watch.py")], { stdio: "ignore" })
      child.on("exit", () => (isWatching = false))
      child.on("error", () => (isWatching = false)) // no python3: helper-reported jobs still show
      child.unref()
    } catch {
      isWatching = false
    }
  }
  startWatcher()
  setInterval(startWatcher, WATCHER_RETRY_MS).unref()

  // The last status of each job, to toast changes; the first poll only records.
  const seen = new Map()
  let isFirst = true
  const poll = async () => {
    for (const job of inScope(load(dir))) {
      const before = seen.get(job.id) ?? (isFirst ? job.status : "running")
      seen.set(job.id, job.status)
      if (before !== job.status && job.status !== "running") {
        await client.tui
          .showToast({ body: { message: summary(job), variant: VARIANT[job.status] } })
          .catch(() => {})
      }
    }
    isFirst = false
  }
  await poll()
  setInterval(() => void poll(), POLL_MS).unref()

  return {
    "shell.env": async (input, output) => {
      if (!input.sessionID) return
      sessions.add(input.sessionID)
      output.env.TRAIN_PROGRESS_SESSION = input.sessionID
    },
    tool: {
      progress: {
        description:
          "List the long-running jobs tracked by train-progress (training runs, `tp run` commands, " +
          "scripts reporting with `tp set`): progress bar, step/total, ETA, status and last output line.",
        args: {},
        async execute(_args, context) {
          const text = await new Promise(resolve =>
            execFile("python3", [path.join(pythonDir(), "tp_view.py"), "--once"], { timeout: 15_000 }, (err, out) =>
              resolve(err ? undefined : out),
            ),
          )
          if (text) return text
          const jobs = load(dir)
          return jobs.length ? jobs.map(summary).join("\n") : `No jobs in ${dir}`
        },
      },
    },
  }
}
