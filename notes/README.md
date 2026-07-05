# Notes Index

This directory is now split between current handoff notes and archived research
history.

## Current Notes

Read these first:

- `tiny_chatbot_v9_broad_336m_current_state.md`: current model card, results,
  failures, commands, artifact policy, and recommended collaboration path
- `NextTODO.md`: short current restart checklist
- `future_codex_chatbot_improvement_note.md`: v9-era improvement direction for
  the next data/evaluation pass
- `current_data_and_artifacts.md`: current corpus rebuild and release artifact
  policy

## Archive

Historical experiments and planning notes live under `notes/archive/`.

The archive explains why the project moved away from older UltraChat-only,
repair-loop, v4-v7, and small baseline paths. Archive notes may mention local
`tmp/` paths or scripts that no longer exist. Treat them as postmortems, not as
current command references.

Start with these archive notes only when you need background:

- `archive/tiny_chatbot_v7_scratch_postmortem.md`
- `archive/tiny_chatbot_v8_direct_answer_plan.md`
- `archive/real_chat_message_importer.md`
- `archive/tiny_chatbot_ultrachat_subword_v3_assistant_loss.md`

## Current Artifact State

The retained local model state is expected under:

```text
tmp/tiny_chatbot_v9_broad_336m_run
```

The final model bundle and tokenizer bundle are kept there locally. Raw datasets,
regenerated corpora, redundant checkpoints, and older run directories were
removed to free disk space and should not be committed.
