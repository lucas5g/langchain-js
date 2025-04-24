import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatGroq } from "@langchain/groq";

const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0
});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});

const vectorStore = new MemoryVectorStore(embeddings);

// Load and chunk contents of blog
const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  // "https://lilianweng.github.io/posts/2023-06-23-agent/",
  'https://defensoria.mg.def.br/defensoria-publica-faz-atendimento-presencial-e-remoto-para-pessoas-interessadas-em-aderir-ao-programa-de-indenizacao-referente-ao-rompimento-da-barragem-de-fundao/',
  {
    selector: pTagSelector
  }
);

const docs = await cheerioLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000, chunkOverlap: 200
});
const allSplits = await splitter.splitDocuments(docs);
// console.log(`Split blog post into ${allSplits.length} sub-documents.`);



// Index chunks
await vectorStore.addDocuments(allSplits)

// Define prompt for question-answering
const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

// const template = `Use the following pieces of context to answer the question at the end.
// If you don't know the answer, just say that you don't know, don't try to make up an answer.
// Use three sentences maximum and keep the answer as concise as possible.
// Always say "thanks for asking!" at the end of the answer.

// {context}

// Question: {question}

// Helpful Answer:`;

// const promptTemplate = ChatPromptTemplate.fromMessages([
//   ["user", template],
// ]);

// Define state for application
const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

// Define application steps
const retrieve = async (state: typeof InputStateAnnotation.State) => {
  const retrievedDocs = await vectorStore.similaritySearch(state.question)
  return { context: retrievedDocs };
};


const generate = async (state: typeof StateAnnotation.State) => {
  const docsContent = state.context.map(doc => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({ question: state.question, context: docsContent });
  const response = await llm.invoke(messages);
  return { answer: response.content };
};


// Compile application and test
const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

// let inputs = { question: 'Qual a empresa irá pagar indenização?' };

let inputs = { question: 'Qual será os dias do multirão?' };

// const result = await graph.invoke(inputs);
// console.log(result.answer);
// console.log(inputs);
// console.log("\n====\n");
// for await (const chunk of await graph.stream(inputs, {
//   streamMode: "updates",
// })) {
//   console.log(chunk);
//   console.log("\n====\n");
// }
const stream = await graph.stream(inputs, { streamMode: "messages" });

for await (const [message, _metadata] of stream) {
  process.stdout.write(message.content);
}