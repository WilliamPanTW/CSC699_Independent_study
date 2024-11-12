import dotenv from "dotenv";
import readline from "readline";
import {GoogleGenerativeAI} from "@google/generative-ai";

dotenv.config();

const genAI = new GoogleGenerativeAI(process.env.API_KEY);

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});


async function run(){
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });
    
    const chat =model.startChat({
        history: [],//start with empty history 
        generationConfig:{
            maxOutputTokens:50 ,
        }
    });
    
    async function askAndRespond(){
        rl.question("You: ",async(msg)=>{
            if(msg.toLowerCase()=="exit"){
                rl.close();
            }else{
                const result = await chat.sendMessage(msg);//added in history 
                const response= await result.response;
                const text=await response.text();
                console.log("AI: ", text);
                askAndRespond();
            }
        });
    }

    askAndRespond();//start message 
}

run();


