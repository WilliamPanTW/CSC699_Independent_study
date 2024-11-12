import dotenv from "dotenv";

dotenv.config();

import {GoogleGenerativeAI} from "@google/generative-ai";


const genAI = new GoogleGenerativeAI(process.env.API_KEY);

async function run(){
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const prompt = "Hi, I am william";
    
    const result = await model.generateContent(prompt);
    const response= await result.response;
    
    console.log(response.text());

}

run();


