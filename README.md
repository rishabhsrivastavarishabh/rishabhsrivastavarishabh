# Sarath AI

**Windows Copilot + Task Automation + Local AI Agent**

Offline capable and focused on OS-level control.

## 🎯 Core Vision

A persistent desktop AI agent that can **listen → plan → execute → notify** across applications with minimal user interaction.

## 🔑 Key Principles

1. **Persistent agent** (always-running background service)
2. **Hybrid intelligence** (local model + cloud model)
3. **Action engine** (task execution, not just chat)
4. **Multilingual voice interface**
5. **Task memory + notification system**

> If the action engine is weak, the product becomes just another chatbot.

## 🧠 High-Level Architecture

### 1) Wake + Voice Layer

- Wake-word detection (for example, "Hey Nova")
- Offline speech recognition
- Multilingual support

**Suggested tech:** Whisper.cpp, Vosk, Coqui STT

### 2) Brain Layer (Hybrid AI)

- **Offline model** for fast local commands:
  - Open applications
  - Check messages
  - Create files
  - Do simple writing
- **Cloud model** for complex reasoning:
  - Draft emails
  - Run research tasks
  - Generate PPT content
  - Handle multi-step workflows

A **router** decides local vs cloud execution based on latency, privacy, cost, and complexity.

### 3) Action Engine (Core Differentiator)

This layer converts intent into real OS-level actions.

Capabilities:

- Open and control apps
- Read notifications
- Automate browser steps
- Create PPT/Word files
- Perform file operations
- Check WhatsApp Web
- Draft emails for approval
- Run automation workflows

**Suggested tech:** Windows UI Automation API, Playwright/Selenium, PowerShell, local tool plugins

### 4) Background Task Manager

Example: *"Check WhatsApp messages and notify me."*

Responsibilities:

- Persistent background workers
- Scheduler and queue management
- State and task memory
- Retry and failure handling
- Safe cancellation and resume

### 5) Notification + Conversation Layer

- Voice responses (TTS)
- Desktop notifications/popups
- Ongoing conversational context
- Safe interruption handling

## ⭐ Realistic Use Cases

- “Open PPT and create slides from this topic.”
- “Check unread WhatsApp messages.”
- “Summarize this PDF.”
- “Write an email and send it after my approval.”
- “Monitor a folder and alert me if a new file appears.”
- “Research this topic while I work.”
- “Fill this form automatically.”
- “Give me a daily briefing when my PC starts.”

## 🧱 MVP Scope

Build first:

1. Wake word
2. Voice-based app launch
3. AI-powered Word/PPT creation
4. One background workflow (WhatsApp Web monitor)
5. Notification system
6. Simple hybrid local/cloud routing

## 🚀 Differentiation

- Offline capable
- Action first (not chat first)
- Persistent desktop agent
- Background workflows
- OS-level automation
- Personal AI operating environment

**Positioning:** *Personal AI Operating Layer*

## ⚠️ Hard Problems

1. Reliable UI automation
2. Offline model resource usage
3. Wake-word accuracy
4. Security and permissions
5. Multilingual voice quality
6. Task-planning reliability
7. Deep Windows integration complexity

## 💰 Future Extensions

- Multi-device agent (PC + mobile)
- Personal memory graph
- Autonomous workflows
- Team agent mode
- Plugin marketplace
- Local private AI workspace
- Developer SDK

## 🧾 One-Line Pitch

“An always-running hybrid AI desktop agent that understands voice, performs real computer tasks in the background, and behaves like a personal operating layer.”

## ✍️ Design Prompt

```text
Design a hybrid AI desktop voice agent that runs persistently on Windows, activates on startup, supports offline and online intelligence routing, executes real OS-level tasks through an action engine, handles background workflows, provides multilingual voice interaction, and notifies users when tasks are completed. Focus on architecture, task planning, action execution, memory, security, and extensibility.
```
