# Tools

This guide provides an overview of essential tools, platforms, and resources for students starting in the AI department.
Computational Resources
1. [Google Colab](https://colab.google.com)

Google Colab is a free cloud-based Jupyter notebook environment that provides access to GPUs.

  GPU Access: Provides approximately 3 hours of free GPU (like Tesla T4) usage per day. This is great for smaller projects and learning.

  Advantages:

        Easy File Editing: You can easily upload, create, and edit files directly within the environment.

        Generative AI Assistance: Includes helpful AI-based coding features to accelerate development.

        Pre-installed Libraries: Most common data science and ML libraries (TensorFlow, PyTorch, etc.) are pre-installed.

  Disadvantages:

        Limited GPU Time: The daily limit of ~3 hours is insufficient for training large models or for extensive projects.

        No Offline Execution: Scripts and tasks cannot be run when you are disconnected. The runtime will disconnect after a period of inactivity.

        Mobile Verification: May require mobile verification to access resources.

2. [Kaggle](kaggle.com)

Kaggle is a platform for data science competitions, datasets, and notebooks.

  GPU Access: Offers a generous quota of 30 hours of GPU time per week. The quota renews every Saturday morning.

  Advantages:

        Larger GPU Quota: Significantly more GPU time than the free tier of Colab.

        Access to Datasets: Hosts a vast collection of public datasets for various tasks.

        Community & Competitions: A great place to learn from others' code and test your skills in competitions.

  Disadvantages:

        No Direct File Editing: You cannot directly edit files within a running notebook session. You must edit them locally and upload a new version of the dataset or script.

        Trick for File Handling: To work around the editing limitation, you can store file content as a string within a code cell and then write that string to a file programmatically during runtime.

3. High-Performance Computing ([HPC](https://github.com/kanha95/HPC-IIT-Delhi)) & [Baadal](https://baadal.iitd.ac.in/baadal) (IIT Delhi)

For more demanding computational needs, IIT Delhi provides access to its own HPC resources.

- HPC: Offers powerful multi-GPU and multi-CPU resources for large-scale experiments. Refer to the official documentation for access and usage: HPC-IIT-Delhi Guide
- Baadal: IITD's private cloud service, which can be used for various computing tasks.

Essential AI & Developer Tools
1. Experiment Tracking: [Wandb](wandb.ai)

Weights & Biases (Wandb) is a tool for tracking your machine learning experiments. It helps you log metrics, visualize results, and compare different runs, which is crucial for research and complex projects.
2. AI-Powered Code Editors

  - [Cursor](cursor.sh): An AI-first code editor designed for pair-programming with AI. It can help you write, debug, and refactor code much faster.
  - [VSCode](code.visualstudio.com): The standard for code editors, with powerful extensions like GitHub Copilot for AI-assisted coding.
  - [Jules](Jules.google.com): Online AI prompt based coding agent.
  - [gemini-cli](github.com/google-gemini/gemini-cli) : Command prompt based coding agent
  - [AI Studio](aistudio.google.com): Free experimental models by google. 

3. Generative AI & LLM Assistants

These tools are invaluable for research, writing, and coding.

    Gemini: Students have been given 1 year free access to pro program
    
    ChatGPT: Excellent for brainstorming, debugging, and generating text/code.

    Perplexity AI: A powerful research and conversational search engine that provides sources for its answers. (Note: A 1-year pro subscription might be available through student packs).

    Claude: Known for its large context window, making it great for summarizing long documents or codebases.

    DeepSeek: Another strong alternative for coding and general queries.

4. Productivity & Note-Taking

    [NotebookLM](notbooklm.google.com): A research assistant from Google that can help you summarize, analyze, and generate insights from your uploaded documents (PDFs, text files, etc.).

    [AI Studio](aistudio.google.com): A platform to explore and build with Google's latest generative AI models.

IITD Specific Resources & Tips
1. Important GitHub Repositories

    IITD-CSE General Guide: https://github.com/ChinmayMittal/IITD-CSE/

    AI Research Fundae: https://github.com/ChinmayMittal/IITD-CSE/tree/main/Misc./Research/Al-Research-Ke-Fundae

2. Connectivity

    Proxy/VPN: You will need to configure the campus proxy or use a VPN for off-campus access to certain resources and research papers.

    TMUX: A terminal multiplexer. It's essential for running long tasks on remote servers (like HPC), as it keeps your session alive even if you disconnect.

3. Student Perks & General Advice

   - Free Software: As a student, you often get free access to Microsoft Office, Adobe products, Swiggy, Github Education and more. T
   - Students get flat 10% discount in *Secular Canteen* near JNU GATE after showing ID card
   - Hospital across the city give IIT students discount (ask whenever you are visiting)
   - Course Planning: Look into "Reading courses of study" to understand course requirements. Create a bucket list of courses you want to take, balancing easier and more challenging ones each semester.
   - Campus Life: Don't forget to take breaks and enjoy campus life! Attend STIC and house dinners.
