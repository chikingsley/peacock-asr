---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

### 🐛 Bug Description  
_A clear and concise description of the issue. What’s broken or not working as expected?_

...

---

### 📂 Affected File(s) or Notebook(s)  
_Provide links to the exact file(s) or notebook(s) where the issue occurs (include the branch name)._

...

---

### ⚙️ Environment Setup  
_Please fill in the relevant details. If not applicable, use `N/A`._

- **Accelerator/GPU**: (e.g. A100, A6000, Mac M1, CPU)  
- **Platform/Host**: (e.g. Local, RunPod, Vast.ai)  
- **Operating System** (`uname -a` or `systeminfo`):  


Optionally, if relevant:
- **Python Version** (`python --version`):  
- **Torch Version** (`python -c "import torch; print(torch.__version__)"`):  
- **CUDA Available** (`python -c "import torch; print(torch.cuda.is_available())"`):  

---

#### 🧪 Transformers Diagnostic (optional)  
If using transformers, run `transformers env` and paste the output below:

<details><summary>Output</summary>

[Paste here]

</details>

---

### 🔁 Minimal Reproduction  
_A minimal and self-contained example that reproduces the issue._

- Keep it short and isolated.
- Include code, public dataset snippet (if applicable), and exact commands.
- Attach logs or terminal output if helpful.

...

---

### ❗Actual Behavior / Error  
_What actually happened? Include full error messages, logs, or stack traces._

<details><summary>Output</summary>

[Paste logs or error messages here]

</details>

---

### ✅ Expected Behavior  
_What did you expect to happen instead?_

...

---

### 🖼️ Screenshots  
_If relevant, add screenshots or gifs that demonstrate the problem._

...

---

### 🧩 Additional Context and Solutions Already Attempted
_Any other context you think might help us debug the issue._

...

---

### 📋 Checklist  
- [ ] Minimal reproduction included  
- [ ] Markdown correctly formatted (use ```code here``` for code blocks)  
- [ ] Logs or error output included if applicable  
- [ ] Sample public datasets included, if applicable  
- [ ] Resolution attempted by feeding the files + this issue description to AI (o3 or Claude Code)
