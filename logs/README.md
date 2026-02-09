# Project Work Logs

This directory contains detailed work logs for tracking development progress, changes, and decisions made throughout the project.

## Purpose

Work logs help us:

- **Track Progress:** Document what work has been completed and what remains
- **Knowledge Sharing:** Help team members understand changes and context
- **Debugging:** Provide a history of changes when troubleshooting issues
- **Documentation:** Create a record of technical decisions and their rationale
- **Onboarding:** Help new team members understand the evolution of the project

## Naming Convention

Use the following naming patterns for log files:

### For Feature Work

```
YYYY-MM-DD_feature-name.md
```

**Examples:**

- `2025-11-29_input-manager-refactor.md`
- `2025-12-01_gesture-recognition-lstm.md`
- `2025-12-05_action-mapper-config.md`

### For Issue/Bug Fixes

```
YYYY-MM-DD_issue-XXX.md
```

**Examples:**

- `2025-11-29_issue-001.md`
- `2025-12-02_issue-042-camera-timeout.md`

### For Research/Exploration

```
YYYY-MM-DD_research-topic.md
```

**Examples:**

- `2025-11-30_research-vit-optimization.md`
- `2025-12-03_research-albef-implementation.md`

## How to Use

### 1. Starting a New Work Session

Copy the template:

```bash
cp logs/LOG_TEMPLATE.md logs/$(date +%Y-%m-%d)_your-feature-name.md
```

Or manually copy `LOG_TEMPLATE.md` and rename it following the naming convention.

### 2. Fill in the Header

Complete the top section with:

- Current date
- Your name
- Branch you're working on
- Related issue number or feature name
- Affected module(s): `input-manager`, `recognition-engine`, `action-mapper`, or `os-controller`

### 3. Update Throughout Your Session

- Add timestamped entries as you make progress
- Document changes, discoveries, and issues in real-time
- Include code snippets when relevant
- Note any architectural decisions or changes

### 4. End of Session Summary

Before you finish:

- Summarize what was accomplished
- Update the tasks checklist
- Document next steps
- Note any blockers or open questions
- List commits made during the session

## Best Practices

### ✅ Do:

- **Be specific:** Include file names, function names, and line numbers when relevant
- **Update frequently:** Add entries as you work, not just at the end
- **Include context:** Explain _why_ you made certain decisions
- **Document failures:** Failed attempts are valuable learning experiences
- **Link resources:** Include links to documentation, Stack Overflow, papers, etc.
- **Note performance impacts:** Document any performance considerations
- **Update commit history:** Keep track of commits made during the session

### ❌ Don't:

- Don't wait until the end to fill out the entire log
- Don't be too vague ("fixed bug" → "fixed camera timeout issue in InputManager.cpp line 45")
- Don't skip documenting challenges or dead ends (they help others avoid the same issues)
- Don't forget to update the tasks checklist as you complete items
- Don't include sensitive information (passwords, API keys, etc.)

## Log Organization

Logs should be kept in this directory with no subdirectories. Use clear, descriptive names so logs are easy to find.

### Archiving Old Logs

After a milestone or release, logs can be archived:

```bash
mkdir -p logs/archive/v1.0
mv logs/2025-* logs/archive/v1.0/
```

## Module Reference

When logging work, specify which module(s) you're working on:

- **input-manager:** Camera capture, frame preprocessing, OpenCV integration
- **recognition-engine:** Hand detection, feature extraction, gesture classification, semantic understanding
- **action-mapper:** Command translation, state management, user configuration
- **os-controller:** Cross-platform execution, keyboard/mouse control, system integration

## Example Log Entry

See `LOG_TEMPLATE.md` for the full template structure. Here's a quick example of a good log entry:

```markdown
### 14:30 - Progress Update

**What I did:**

- Refactored InputManager to use thread-safe frame buffering
- Added smart mirroring detection to handle different camera orientations

**Changes made:**

- Files modified: `src/input-manager/src/InputManager.cpp`, `src/input-manager/include/InputManager.hpp`
- Added new method: `detectMirrorState()` at line 78

**Observations/Discoveries:**

- Some MacBook webcams report frames already mirrored, causing double-flip issue
- Solution: Check camera metadata before applying flip

**Issues encountered:**

- Initial implementation caused race condition in multi-threaded access

**Solutions applied:**

- Added std::mutex to protect frame buffer (line 45-52)
- Implemented atomic flag for camera state checking
```

## Questions?

If you have questions about the logging practice or suggestions for improving the template, please discuss with the team or create an issue.

---

**Remember:** Good logs make better developers and stronger teams! 🚀
