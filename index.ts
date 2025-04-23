// src/index.ts
import fs from "fs/promises";
import path from "path";
import { ChatGroq } from "@langchain/groq";
import { Document } from "langchain/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { formatDocumentsAsString } from "langchain/util/document";

const model = new ChatGroq({
  model: "meta-llama/llama-4-scout-17b-16e-instruct",
  apiKey: process.env.GROQ_API_KEY!,
  temperature: 0.1,
});

async function loadVectorStoreFromJSON(filePath: string) {
  const file = await fs.readFile(filePath, "utf-8");
  const jsonData = JSON.parse(file);

  const docs = jsonData.map((item: any) => {
    return new Document({
      pageContent: item.text,
      metadata: item.metadata,
    });
  });

  const embedder = new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2",
  });

  const store = await MemoryVectorStore.fromDocuments(docs, embedder);
  return store;
}

async function main() {
  const store = await loadVectorStoreFromJSON(path.resolve("vectorstore.json"));
  const retriever = store.asRetriever();

  const chain = RunnableSequence.from([
    async (input: string) => {
      const relevantDocs = await retriever.getRelevantDocuments(input);
      return formatDocumentsAsString(relevantDocs);
    },
    model,
    new StringOutputParser(),
  ]);

  const question = "Tem quantas inscrições no barreiro?";
  const answer = await chain.invoke(question);

  console.log("❓ Pergunta:", question);
  console.log("💬 Resposta:", answer);
}

main();
