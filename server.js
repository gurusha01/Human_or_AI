const express = require('express');
const fs = require('fs');
const bodyParser = require('body-parser');
const path = require('path'); // Ensure path is imported
const app = express();

app.use(bodyParser.json()); // To parse JSON request bodies
app.use(bodyParser.urlencoded({ extended: true }));

// Serve static files from the current directory
app.use(express.static(__dirname)); // This will serve all files in the main folder

// Serve the index.html directly from the root directory
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html')); // Use path to serve the HTML file
});

// Endpoint to save the user's answer
app.post('/save-answer', (req, res) => {
    const { answer, isCorrect } = req.body;
    const logEntry = `Answer: ${answer}, Correct: ${isCorrect}, Time: ${new Date().toISOString()}\n`;

    // Append the data to a file
    fs.appendFile('answers.txt', logEntry, (err) => {
        if (err) {
            return res.status(500).send('Error saving the answer');
        }
        res.status(200).send('Answer saved successfully');
    });
});

const PORT = 3000; // You can change this to another port if needed
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});