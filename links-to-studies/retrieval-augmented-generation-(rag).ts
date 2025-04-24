import { ChatOpenAI } from "@langchain/openai";

// Define a system prompt that tells the model how to use the retrieved context
const systemPrompt = `You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Context: {context}:`;

// Define a question
const question =
  "What are the main components of an LLM-powered autonomous agent system?";

// Retrieve relevant documents
const docs = await retriever.invoke(question);

// Combine the documents into a single string
const docsText = docs.map((d) => d.pageContent).join("");

// Populate the system prompt with the retrieved context
const systemPromptFmt = systemPrompt.replace("{context}", docsText);

// Create a model
const model = new ChatOpenAI({
  model: "gpt-4o",
  temperature: 0,
});

// Generate a response
const questions = await model.invoke([
  {
    role: "system",
    content: systemPromptFmt,
  },
  {
    role: "user",
    content: question,
  },
]);