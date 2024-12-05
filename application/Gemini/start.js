import dotenv from "dotenv";

dotenv.config({ path: 'D:\\CSC699_Independent_study\\application\\Gemini\\dot.env' });

import {GoogleGenerativeAI} from "@google/generative-ai";


const genAI = new GoogleGenerativeAI(process.env.API_KEY);
async function run(){
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const prompt = "what is the answer to life the universe and everything?";
    
    const result = await model.generateContent(prompt);
    const response= await result.response;
    
    console.log(response.text());

}

run();


