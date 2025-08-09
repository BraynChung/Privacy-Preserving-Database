# 🛡️ Privacy-Preserving Database: A Modern Data Privacy Playground

Welcome to the **Privacy-Preserving Database** project! This is your all-in-one, hands-on laboratory for exploring, benchmarking, and understanding the real-world tradeoffs between privacy and utility in data analytics. Whether you’re a privacy researcher, a data scientist, or just privacy-curious, this project is designed to help you experiment with and compare state-of-the-art privacy-preserving techniques on real data.

---

## 🚀 What’s Inside?

- **Differential Privacy System**: Run SQL queries with rigorous privacy guarantees using Laplace, InstMean, and Holistic mechanisms. Track privacy budgets, noisy results, and accuracy loss.
- **AES Encryption System**: Encrypt and decrypt sensitive columns, then benchmark the impact on query performance and utility.
- **K-Anonymity System**: Anonymize datasets using k-anonymity, generalization, and suppression. Measure how much utility you retain as you increase privacy.
- **Performance Assessment Suite**: Automated, repeatable benchmarking for all privacy mechanisms. Get detailed reports on latency, CPU usage, privacy budget consumption, and utility loss.
- **Extensive Test Suite**: Scripts and utilities for quick checks, diagnostics, and regression testing.

---

## 🏗️ Project Structure

```
Privacy-Preserving-Database/
├── config/           # Database configs and connection helpers
├── dataset/          # Example datasets (e.g., UCI Adult)
├── docs/             # Documentation, implementation notes, and structure
├── src/              # Core source code for all privacy systems
│   ├── aes_system.py
│   ├── dp_system.py
│   ├── k_anonymity_system.py
│   ├── performance_assessment.py
│   └── ...
├── test/             # Test scripts and quick performance checks
├── README.md         # (You are here!)
└── ...
```

---

## 🧑‍💻 How to Use

1. **Install Requirements**
	- Python 3.10+
	- `pip install -r requirements.txt` (if provided)
	- Ensure you have SQLite (default) or configure your own DB in `config/db_config.py`

2. **Run All Performance Tests**
	```bash
	python src/performance_assessment.py
	```
	This will:
	- Run comprehensive tests for Differential Privacy, AES Encryption, and K-Anonymity
	- Save detailed JSON results and print human-readable reports

3. **Explore Results**
	- Check the generated `*_performance_results_*.json` files for raw data
	- Read the console output for summary reports
	- Tweak parameters in `src/performance_assessment.py` to experiment with different settings

4. **Add Your Own Queries or Datasets**
	- Edit `DEFAULT_TEST_QUERIES` in `src/dp_system.py` to add new SQL queries
	- Drop your own CSVs into `dataset/` and update paths as needed

---

## 📊 What Can You Learn?
- How much accuracy do you lose as you increase privacy?
- How do different privacy mechanisms compare in speed and utility?
- What’s the real cost (in latency, CPU, and accuracy) of privacy for your data?
- How does k-anonymity stack up against differential privacy?

---

## 🛠️ Key Files
- `src/dp_system.py` — Differential Privacy engine
- `src/aes_system.py` — AES encryption/decryption
- `src/k_anonymity_system.py` — K-anonymity anonymizer
- `src/performance_assessment.py` — Benchmarking and reporting
- `test/` — Quick tests and diagnostics
- `dataset/adult.csv` — Example dataset

---

## 🤝 Contributing
Pull requests, issues, and suggestions are welcome! This project is a learning tool and a research platform—feel free to fork, extend, and experiment.

---

## 📚 Further Reading
- [docs/DIRECTORY_STRUCTURE.md](docs/DIRECTORY_STRUCTURE.md)
- [docs/IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md)
- [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

---

## 📝 License
This project is for educational and research purposes. See LICENSE for details.

---

## 🌟 Acknowledgements
Thanks to all privacy researchers and open-source contributors who inspire and enable privacy innovation!

---

**Happy experimenting!**

