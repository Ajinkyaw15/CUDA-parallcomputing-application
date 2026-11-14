# Distributed Lock System (ZooKeeper + Java)

## 📌 Overview
This project demonstrates **Distributed Mutual Exclusion** using:
1. Apache ZooKeeper (Industry-level implementation)
2. Java File-based Lock (Custom implementation)

The goal is to show how **only one process can access a shared resource at a time**.

---

## 🧠 Concept
A **Distributed Lock** ensures:
- Only one client holds the lock
- Other clients are denied access
- Prevents race conditions

This concept is widely used in systems like:
- Google Chubby
- Bigtable
- Distributed Databases

---

## ⚙️ Technologies Used
- Java
- Apache ZooKeeper

---

## 🧪 Implementation 1: ZooKeeper Lock

### Steps:
1. Start ZooKeeper server
2. Open ZooKeeper CLI
3. Run:

create /lock "client1"
create /lock "client2"


### Output:
- First command → Lock acquired
- Second command → `Node already exists`

---

## 📸 ZooKeeper Demo

![ZooKeeper Demo](zookeeper-demo.png)

---

## 💻 Implementation 2: Java Lock

### Code Logic:
- Create a file `lockfile.txt`
- If file is created → Lock acquired
- If file exists → Lock denied

### Run:

javac DistributedLock.java
java DistributedLock


### Output:
- Terminal 1 → Lock acquired
- Terminal 2 → Lock already acquired

---

## 🔄 Comparison

| Feature | ZooKeeper | Java Implementation |
|--------|----------|--------------------|
| Type | Distributed | Local |
| Lock Mechanism | Znode (/lock) | File (lockfile.txt) |
| Fault Tolerance | Yes | No |
| Real-world Usage | Yes | Simulation |

---

## 🎯 Conclusion
- ZooKeeper provides **reliable distributed locking**
- Java implementation simulates the same concept locally
- Both ensure **mutual exclusion**

## 📁 Project Structure


DS/
├── DistributedLock.java
├── README.md
├── zookeeper-demo.png


## 👨‍💻 Author
Ajinkya Wadhale
