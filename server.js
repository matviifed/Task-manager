const express = require("express");
const cors = require("cors");
const { MongoClient, ObjectId } = require("mongodb");

const app = express();
app.use(cors());
app.use(express.json());

const mongoURL = "mongodb://localhost:27017";
const dbName = "task-manager";
let db;

// MongoDB connection
async function connectDB() {
    try {
        const client = new MongoClient(mongoURL);
        await client.connect();
        db = client.db(dbName);
        console.log("Connected to MongoDB");
    } catch (error) {
        console.error("MongoDB connection error:", error);
        process.exit(1);
    }
}

connectDB();

// API Routes
app.get("/tasks", async (req, res) => {
    try {
        const tasks = await db.collection("tasks").find({}).toArray();
        res.json(tasks);
    } catch (error) {
        console.error("Error fetching tasks:", error);
        res.status(500).json({ message: "Error fetching tasks" });
    }
});

app.post("/tasks", async (req, res) => {
    try {
        const { title, description, priority, deadline } = req.body;

        if (!title) {
            return res.status(400).json({ message: "Task title is required" });
        }

        const newTask = {
            title,
            description: description || "",
            priority,
            deadline: deadline ? new Date(deadline) : null,
            status: "todo",
            createdAt: new Date()
        };

        const result = await db.collection("tasks").insertOne(newTask);
        newTask._id = result.insertedId;

        res.status(201).json(newTask);
    } catch (error) {
        console.error("Error creating task:", error);
        res.status(500).json({ message: "Error creating task" });
    }
});

app.delete("/tasks/:id", async (req, res) => {
    try {
        const { id } = req.params;
        if (!ObjectId.isValid(id)) {
            return res.status(400).json({ message: "Invalid task ID" });
        }
        
        const result = await db.collection("tasks").deleteOne({ 
            _id: new ObjectId(id) 
        });

        if (result.deletedCount === 0) {
            return res.status(404).json({ message: "Task not found" });
        }

        res.json({ message: "Task deleted successfully" });
    } catch (error) {
        console.error("Error deleting task:", error);
        res.status(500).json({ message: "Error deleting task" });
    }
});

app.post("/completed-tasks", async (req, res) => {
    try {
        const { id, title, description, priority, deadline } = req.body;

        if (!id) {
            return res.status(400).json({ message: "Task ID is required" });
        }

        const completedTask = {
            originalId: id,
            title,
            description,
            priority,
            completedDate: new Date(),
        };

        await db.collection("completed-tasks").insertOne(completedTask);
        await db.collection("tasks").deleteOne({ _id: new ObjectId(id) });

        res.json({ message: "Task moved to completed", completedTask });
    } catch (error) {
        console.error("Error completing task:", error);
        res.status(500).json({ message: "Error completing task" });
    }
});

app.get("/completed-tasks", async (req, res) => {
    try {
        const completedTasks = await db.collection("completed-tasks").find({}).toArray();
        res.json(completedTasks);
    } catch (error) {
        console.error("Error fetching completed tasks:", error);
        res.status(500).json({ message: "Error fetching completed tasks" });
    }
});

app.delete("/completed-tasks/:id", async (req, res) => {
    try {
        const { id } = req.params;
        if (!ObjectId.isValid(id)) {
            return res.status(400).json({ message: "Invalid task ID" });
        }

        const result = await db.collection("completed-tasks").deleteOne({ 
            _id: new ObjectId(id) 
        });

        if (result.deletedCount === 0) {
            return res.status(404).json({ message: "Completed task not found" });
        }

        res.json({ message: "Completed task deleted" });
    } catch (error) {
        console.error("Error deleting completed task:", error);
        res.status(500).json({ message: "Error deleting completed task" });
    }
});

app.delete("/completed-tasks", async (req, res) => {
    try {
        await db.collection("completed-tasks").deleteMany({});
        res.json({ message: "All completed tasks cleared" });
    } catch (error) {
        console.error("Error clearing completed tasks:", error);
        res.status(500).json({ message: "Error clearing completed tasks" });
    }
});

// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});