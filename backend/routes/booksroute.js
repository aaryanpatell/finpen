import express from "express";
import { Book } from "../models/bookModel.js";

const router = express.Router();

/* post method to add a book */
router.post('/', async (request, response) => {
    try {
        if (!request.body.title || !request.body.author || !request.body.publishYear) {
            return response.status(400).send({
                message: "SEND ALL REQUIRED FIELDS: TITLE, AUTHOR, PUBLISHYEAR"
            });
        }
        const newBook = {
            title: request.body.title,
            author: request.body.author,
            publishYear: request.body.publishYear,
        };

        const book = await Book.create(newBook);

        return response.status(201).send(book);
    } catch (error) {
        console.log(error.message);
        response.status(500).send({ message: error.message });
    }
});

/* get method for all books */
router.get('/', async (request, response) => {
    try {
        const books = await Book.find({});

        return response.status(200).json({
            count: books.length,
            data: books
        });
    } catch (error) {
        console.log(error.message);
        response.status(500).send({ message: error.message });
    }
})

/* get method for book by id */
router.get('/:id', async (request, response) => {
    try {
        const { id } = request.params;
        const book = await Book.findById(id);

        return response.status(200).json(book);
    } catch (error) {
        console.log(error.message);
        response.status(500).send({ message: error.message });
    }
})

/* put method to update a book */
router.put('/:id', async (request, response) => {
    try {
        if (!request.body.title || !request.body.author || !request.body.publishYear) {
            return response.status(400).send({
                message: "SEND ALL REQUIRED FIELDS: TITLE, AUTHOR, PUBLISHYEAR"
            });
        }

        const { id } = request.params;
        const result = await Book.findByIdAndUpdate(id, request.body);

        if (!result) {
            return response.status(404).json({ message: "BOOK NOT FOUND" });
        }

        return response.status(200).json({ message: "BOOK UPDATED SUCCESSFULLY", result });
    } catch (error) {
        console.log(error.message);
        response.status(500).send({ message: error.message });
    }
})

/* delete method to delete book */
router.delete('/:id', async (request, response) => {
    try {
        const { id } = request.params;

        const result = await Book.findByIdAndDelete(id);

        if (!result) {
            return response.status(404).json({ message: "BOOK NOT FOUND" });
        }

        return response.status(200).send({ message: "BOOK DELETED SUCCESSFULLY" })

    } catch (error) {
        console.log(error.message);
        response.status(500).send({ message: error.message });
    }
});

export default router;