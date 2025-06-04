const express = require("express");
const cors = require("cors");
const { MongoClient, ObjectId } = require("mongodb");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const path = require("path");
const fs = require("fs");
const multer = require("multer");
const { spawn, exec } = require("child_process");
require('dotenv').config();

if (!process.env.GEMINI_API_KEY) {
    process.env.GEMINI_API_KEY = "AIzaSyCI1j_59ODIK9tlgMjOGSUMFHExmi1e6cw"; // Replace with your actual API key
}

if (!process.env.GEMINI_API_KEY) {
    console.error("GEMINI_API_KEY is not set in environment variables");
    process.exit(1);
}
const app = express();
app.use(cors());
app.use(express.json());

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);


let fetchSetup;
async function setupFetch() {
    if (!fetchSetup) {
        const { default: fetch, Headers } = await import('node-fetch');
        global.fetch = fetch;
        global.Headers = Headers;
        fetchSetup = true;
    }
}

const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

app.use('/uploads', express.static(uploadDir));

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });

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

// GPT integration
app.post("/tasks/tips", async (req, res) => {
    try {
        await setupFetch();
        
        const { title, description } = req.body;

        if (!title) {
            return res.status(400).json({ message: "Task title is required" });
        }

        const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

        const prompt = `Give me 3-5 practical tips for completing this task: "${title}". ${description ? `Additional context: ${description}` : ''}. Use the language that "${title}" has.`;

        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text();

        res.json({ tips: text });
    } catch (error) {
        console.error("Error getting tips:", error);
        res.status(500).json({ 
            message: "Error getting tips", 
            error: error.message 
        });
    }
});


app.get("/images", async (req, res) => {
    try {
        const images = await db.collection("images").find({}).toArray();
        res.json(images);
    } catch (error) {
        console.error("Error fetching images:", error);
        res.status(500).json({ message: "Error fetching images" });
    }
});

app.post("/upload_image", upload.single('image'), async (req, res) => {
    if (!req.file) return res.status(400).json({ success: false, error: 'No image file provided' });

    try {
        const imagePath = req.file.path;
        console.log("Image uploaded:", req.file.originalname, imagePath);

        const newImage = {
            filename: req.file.filename,  // Use stored filename
            path: `/uploads/${req.file.filename}`,  // Relative path for frontend
            uploadDate: new Date(),
            size: req.file.size
        };

        const result = await db.collection("images").insertOne(newImage);

        res.json({
            success: true,
            imageId: result.insertedId,
            filename: req.file.filename
        });

    } catch (error) {
        console.error("Error saving image:", error);
        res.status(500).json({ success: false, error: "Error saving image" });
    }
});

app.delete("/images/:id", async (req, res) => {
    try {
        const { id } = req.params;
        if (!ObjectId.isValid(id)) return res.status(400).json({ message: "Invalid image ID" });

        const result = await db.collection("images").deleteOne({ _id: new ObjectId(id) });

        if (result.deletedCount === 0) return res.status(404).json({ message: "Image not found" });

        res.json({ message: "Image deleted successfully" });
    } catch (error) {
        console.error("Error deleting image:", error);
        res.status(500).json({ message: "Error deleting image" });
    }
});


app.post("/analyze_image", async (req, res) => {
    const { imageId } = req.body;
    if (!imageId) {
        return res.status(400).json({ success: false, error: "No image ID provided" });
    }

    try {
        const imageDoc = await db.collection("images").findOne({ _id: new ObjectId(imageId) });
        if (!imageDoc) {
            return res.status(404).json({ success: false, error: "Image not found" });
        }

    
        const imagePath = path.join(__dirname, "uploads", imageDoc.filename);

    
        const pythonProcess = spawn('python', ['image_analyzer.py', imagePath]);

        let output = '';
        let errorOutput = '';
        
        pythonProcess.stdout.on('data', (data) => {
            const chunk = data.toString();
            console.log("Python stdout:", chunk);
            output += chunk;
        });
        
        pythonProcess.stderr.on('data', (data) => {
            const chunk = data.toString();
            console.log("Python stderr:", chunk);
            errorOutput += chunk;
        });
        
        pythonProcess.on('close', (code) => {
            console.log(`Python process exited with code ${code}`);
            console.log("Final output:", output);
            
            let description = output;
            const startMarker = "--- DESCRIPTION START ---";
            const endMarker = "--- DESCRIPTION END ---";
            
            const startIdx = output.indexOf(startMarker);
            const endIdx = output.indexOf(endMarker);
            
            if (startIdx !== -1 && endIdx !== -1) {
                description = output.substring(startIdx + startMarker.length, endIdx).trim();
            }
            
            if (code !== 0 || !description) {
                console.error('Python script error or no output');
                return res.status(500).json({ 
                    success: false, 
                    error: "Failed to analyze image",
                    details: errorOutput,
                    pythonOutput: output
                });
            }
            
            res.json({ success: true, description: description });
        });

    } catch (error) {
        console.error("Error in image analysis route:", error);
        res.status(500).json({ 
            success: false, 
            error: "Internal server error during image analysis" 
        });
    }
});

function generateFallbackDescription(filename) {
    const name = path.basename(filename, path.extname(filename));
    const words = name.split(/[-_\s.]/g).filter(word => word.length > 0);
    
  
    const capitalizedWords = words.map(word => 
        word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
    );
    
    if (capitalizedWords.length === 0) {
        return "Stylish product with a modern design. Perfect for any occasion.";
    }
    
    let productType = "item";
    if (filename.toLowerCase().includes("shirt") || filename.toLowerCase().includes("tshirt")) {
        productType = "T-shirt";
    } else if (filename.toLowerCase().includes("pant") || filename.toLowerCase().includes("trouser")) {
        productType = "pants";
    } else if (filename.toLowerCase().includes("shoe") || filename.toLowerCase().includes("sneaker")) {
        productType = "footwear";
    } else if (filename.toLowerCase().includes("hat") || filename.toLowerCase().includes("cap")) {
        productType = "headwear";
    }
    
    return `Stylish ${productType} featuring ${capitalizedWords.join(' ')} design. Great addition to any wardrobe.`;
}


// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});