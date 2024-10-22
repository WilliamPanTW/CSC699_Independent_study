const express = require('express');
const app = express();
const PORT = 8080;

app.post('/milkshake/:id', (req, res) => {
    const { id } =req.params;
    res.status(200).send(`Milkshake with ID ${id} received!`);
});

// fire it up by listen to a port
app.listen(PORT, () => console.log(`API running at http://localhost:${PORT}`));
