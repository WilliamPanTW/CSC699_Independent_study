import dotenv from "dotenv";
import * as fs from "fs";
import mime from 'mime-types'

dotenv.config();

import {GoogleGenerativeAI} from "@google/generative-ai";


const genAI = new GoogleGenerativeAI(process.env.API_KEY);

// image path and type 
function fileToPart(filePath){
    const mimeType = mime.lookup(filePath);//token
    return{
        inlineData:{
            data: Buffer.from(fs.readFileSync(filePath)).toString("base64"),
            mimeType,
        }, 
    };
}

async function run(){
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });
    const prompt = "what is the scene number?";
    
    const filePath = 'D:\\CSC699_Independent_study\\application\\map\\test.jpg';
    const imageParts = [fileToPart(filePath)]

    // ... meaning Merging objects == [prompt,data,mimeType]
    const result = await model.generateContent([prompt,...imageParts]);
    const response= await result.response;
    const text=response.text()
    console.log(text);

}

run();
