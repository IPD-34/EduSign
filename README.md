# EduSign

EduSign is a bidirectional sign language translation system designed for higher education. It enables real-time communication between deaf and hearing individuals through continuous sign recognition, speech-to-sign translation, and a shared, course-specific technical vocabulary. Unlike generic translators, EduSign continuously learns and propagates technical signs across users, making STEM communication more accurate and accessible.

---

## Motivation

Existing sign language translation systems primarily target everyday communication and struggle with technical academic terminology.

In classrooms, laboratories, office hours, and project discussions, interpreters may be unavailable or unfamiliar with subject-specific vocabulary. Technical signs are often created temporarily and never shared beyond a single interaction.

EduSign addresses these challenges by introducing **course-aware**, **context-aware**, and **collaborative** translation that allows technical vocabulary to evolve and be reused across users.

---

## ✨ Core Features

### 📚 Community Vocabulary Propagation
- Shared course-specific technical vocabulary repository.
- Unknown technical terms require confirmation only once.
- Confirmed signs become available to future users automatically.
- Conflict resolution for multiple signs representing the same concept.

### 🎓 Context-Aware Translation
- Uses course-specific vocabulary during translation.
- Improves recognition of technical terminology.
- Produces more accurate translations than generic models.

### 🤖 End-to-End Continuous Sign Recognition
- Single transformer architecture for joint segmentation and recognition.
- Operates on MediaPipe Holistic landmark sequences.
- Fully transformer-based pipeline without CNNs or GCNs.

### 📊 Confidence-Aware Translation
- Generates confidence scores for every prediction.
- Uses shared vocabulary when confidence is low.
- Requests clarification instead of producing incorrect translations.

### 👥 Multi-Speaker Academic Support
- Supports lectures, tutorials, laboratories, and group discussions.
- Speaker attribution preserves conversational structure.

---

## ⚙️ System Overview

### Sign → Text / Speech

1. Capture sign language using a webcam.
2. Extract MediaPipe Holistic landmarks.
3. Perform joint sign segmentation and recognition.
4. Estimate prediction confidence.
5. Resolve technical terms using the shared vocabulary.
6. Generate translated text or speech.

### Speech / Text → Sign

1. Capture speech or typed input.
2. Convert speech to text.
3. Identify the active speaker.
4. Convert text into sign gloss sequences.
5. Retrieve course-specific technical signs.
6. Render translated sign output.

---

## 🚀 What Makes EduSign Different?

| Existing Systems | EduSign |
|------------------|----------|
| Generic translation | Course-aware translation |
| Static vocabulary | Dynamic shared vocabulary |
| Individual dictionaries | Community-maintained vocabulary |
| Repeated handling of unknown terms | One-time confirmation reused by all users |
| One-to-one conversations | Multi-speaker academic conversations |
| Separate segmentation & recognition | Unified end-to-end transformer |
| Translation only | Translation + collaborative vocabulary learning |

---

## Vision

EduSign aims to make higher education more accessible by creating an intelligent sign language translation platform that improves over time. Through collaborative vocabulary learning, context-aware translation, continuous sign recognition, and multi-speaker support, EduSign builds a growing repository of technical signs that benefits future learners while addressing the communication challenges of academic environments.
