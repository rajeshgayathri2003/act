# .agents

`.agents/` is a convention created by Marco ([@Ocramaru](https://github.com/Ocramaru)) for keeping coding-agent notes next to the code they describe. If you don't work with a coding agent, or this isn't a convention you want to keep, you can ignore this folder: it only holds extra context for agents wired to read it.

Any directory can hold an `.agents/` folder. The notes travel with the code through git and are scoped to their directory: one higher up the tree covers more ground, and one deeper in narrows or overrides it.

| File | Holds |
|---|---|
| `context.md` | what this directory assumes and how its parts fit together: the context needed to change it correctly |
| `learned.md` | what experiments showed and the traps that already cost time, so they are only paid once |
| `todo.md` | open work for this directory |
| `todo.archive.md` | finished work, kept as the record of why the code looks the way it does |
| `standards/` | this repo's additions to the shared style rules, such as how Python or commits are written here |
| `agents.toml` | settings for the tooling in this directory |

A note that gets long can be split into a folder of the same name, one file per section, with the original kept as a short index of the parts:

```
.agents/
  learned.md          the index
  learned/
    traps.md          one section
    settled.md        another
```

`.reviews/` holds a local history of code reviews and is never committed.

## How the notes reach an agent

The tooling is one more `.agents`, installed once per machine at `~/.agents` from [Ocramaru/.agents](https://github.com/Ocramaru/.agents). It holds the shared standards and the hooks. When an agent edits a file here, the hooks find every `.agents/` above that file and hand the agent the notes and standards that govern it: `context.md` in full, and the other notes by name so it can open them when it needs to. Claude Code gets this through its hooks; Codex, Gemini and other agents are pointed at the same files through their instructions file.

Without the tooling, these are plain markdown and read fine.

## Using it

To have your own agent read these notes:

```sh
git clone https://github.com/Ocramaru/.agents ~/.agents
~/.agents/bin/agent install
```

That wires up every agent it finds on the machine, with a few starter standards. Your own standards, reviewers and settings can live in a private profile repo layered on top, and they never land in a shared repo like this one; the [`.agents` README](https://github.com/Ocramaru/.agents#your-profile) shows how.
