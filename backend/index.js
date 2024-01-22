import express from "express";
import { PORT, mongoDBURL } from "./config.js";
import mongoose from "mongoose";
import booksRoute from "./routes/booksroute.js"

const app = express();

app.use(express.json());

app.get('/', (request, response) => {
    console.log("HELLO");
    return response.status(234).send("WELCOME TO MERN STACK");
});

app.use('/books', booksRoute);

mongoose
    .connect(mongoDBURL)
    .then(() => {
        console.log("APPLICATION CONNECTED TO DB");
        app.listen(PORT, () => {
            console.log("APPLICATION ACTIVE AND LISTENING");
        });
    })
    .catch((error) => {
        console.log(error);
    });