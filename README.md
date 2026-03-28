# 🫀 Cardiotect - Automated Coronary Calcium Scoring

**A deep learning system for automated coronary calcium scoring from CT scans.**

This project uses artificial intelligence to detect and score calcium deposits in coronary arteries, helping with early detection of heart disease. **You don't need any programming experience to use this!**

---

## 📥 **For Complete Beginners - How to Get This Project**

### **Step 1: Download the Project**
1. **Click the green "Code" button** at the top of this GitHub page
2. **Select "Download ZIP"** from the dropdown menu
3. **Save the ZIP file** to your Desktop (or any folder you like)
4. **Extract the ZIP file** by right-clicking it and selecting "Extract All"
5. **Open the extracted folder** - you should see files like `start_cardiotect_web.bat`, `README.md`, etc.

### **Step 2: Install Required Software**
You need **Python** and **Git LFS** (for large model files). Here's how:

1. **Install Python 3.8 or newer**:
   - Go to [python.org/downloads](https://www.python.org/downloads/)
   - Download the latest version for Windows
   - **IMPORTANT**: During installation, check "Add Python to PATH"
   - Click "Install Now"

2. **Install Git (optional, for updates)**:
   - Go to [git-scm.com](https://git-scm.com/)
   - Download and install with default settings
   - This allows you to get updates easily

### **Step 3: Run the Application**
1. **Navigate to the extracted folder**
2. **Double-click `start_cardiotect_web.bat`**
3. **Wait** - The first time will take 5-10 minutes to:
   - Create a virtual environment
   - Install all required packages
   - Download necessary components
4. **The web interface will open automatically** in your default browser

**That's it!** You're now ready to use Cardiotect.

---

## 🖥️ **How to Use Cardiotect**

### **Basic Workflow:**
1. **Launch the application** using `start_cardiotect_web.bat`
2. **Upload a CT scan** (DICOM format) using the web interface
3. **Click "Run Analysis"** - The AI will process the scan
4. **View results** - Calcium scores and visualizations appear automatically

### **What You Need:**
- **CT scan files** in DICOM format (.dcm files)
- **Patient folder** containing the DICOM files
- **Optional**: XML annotation files for ground truth comparison

---

## 📁 **Dataset Configuration (Optional)**

**The medical dataset is NOT included** (for privacy reasons). If you have your own dataset:

1. **Edit the `.env` file** in the project root
2. **Set your dataset path**:
   ```
   CARDIOTECT_DATASET_ROOT=C:\path\to\your\dataset
   ```
3. **Ensure your dataset follows this structure**:
   ```
   your_dataset/
   └── patient_scans/
       ├── patient_001/
       │   ├── scan_001.dcm
       │   └── scan_002.dcm
       └── patient_002/
           └── scan_001.dcm
   ```

---

## 🤖 **Model Checkpoints**

**Pre-trained AI models are included!** You don't need to download anything extra.

- **Location**: `outputs/checkpoints/`
- **Files**: `best.ckpt` and `latest.ckpt`
- **How they work**: The application automatically loads these when you start it
- **No setup required** - they work out of the box

---

## 🛠️ **Troubleshooting Common Issues**

### **"Python not found" error**
- Reinstall Python and **check "Add Python to PATH"**
- Restart your computer after installation

### **"Checkpoint not found" error**
- Make sure you downloaded the **complete ZIP file**
- Don't move individual files - keep the folder structure intact
- Try re-downloading and extracting the ZIP

### **"Module not found" errors**
- Run `setup.bat` manually (in the `extras` folder)
- Or delete the `venv` folder and run `start_cardiotect_web.bat` again

### **Web interface doesn't open**
- Check if port 8765 is blocked by firewall
- Try running as administrator
- Look for error messages in the command window

### **Need more help?**
- **GitHub Issues**: Go to the "Issues" tab on this GitHub page
- **Contact**: reuvenmundala12@gmail.com

---

## 📂 **Project Structure Explained**

```
Cardiotect/
├── README.md                 # This instruction file
├── start_cardiotect_web.bat  # MAIN FILE - Double-click to start!
├── .env.example              # Configuration template
├── outputs/
│   └── checkpoints/          # AI models (included)
└── extras/                   # All other files
    ├── cardiotect_cac/       # Core AI code
    ├── web_gui/              # Web application
    ├── gui_v2/               # Desktop version 2
    ├── gui_v3/               # Desktop version 3
    ├── setup.bat             # Setup script (auto-runs)
    └── requirements.txt      # Python packages list
```

---

## 💻 **System Requirements**

- **Operating System**: Windows 10 or 11
- **Python**: 3.8 or newer
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space
- **GPU**: NVIDIA GPU with CUDA (optional, but makes analysis much faster)

---

## 🔄 **Getting Updates**

If you installed Git:
1. Open Command Prompt in the project folder
2. Run: `git pull`
3. Run: `git lfs pull` (to update models if changed)

---



---

## 🆘 **Need Help?**

1. **Check the troubleshooting section above**
2. **Look at the "Issues" tab** on this GitHub page
3. **Contact**: reuvenmundala12@gmail.com

---

**🫀 Thank you for using Cardiotect! Together, we can advance cardiovascular disease detection.**