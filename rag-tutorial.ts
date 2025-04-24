/**
 * CHAT MODEL
 */

import { ChatOpenAI } from "@langchain/openai";

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0
});

/**
 * EMBEDDINGS MODEL
 */
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});

/**
 * VECTOR STORE
 */
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const vectorStore = new MemoryVectorStore(embeddings);


/**
 * LOADING DOCUMENTS
 */
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
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
console.assert(docs.length === 1);
// console.log(`Total characters: ${docs[0].pageContent.length}`);
// console.log(docs[0].pageContent.slice(0, 50));

/**
 * SPLITTING documents
 */
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000, chunkOverlap: 200
});
const allSplits = await splitter.splitDocuments(docs);

// console.log(`Split blog post into ${allSplits.length} sub-documents.`);

/**
 * Storing documents
 */
// Index chunks
await vectorStore.addDocuments(allSplits)

/**
* RETRIEVAL AND GENERATION PROMPT
*/
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
// Define prompt for question-answering
// const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

/**
 * CUSTOM PROMPT
 */

const template = `Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:`;

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["user", template],
]);


/**
 * STATE
 */
import { Document } from "@langchain/core/documents";
import { Annotation } from "@langchain/langgraph";
// Define state for application
const InputStateAnnotation = Annotation.Root({
  question: Annotation<string>,
});

const StateAnnotation = Annotation.Root({
  question: Annotation<string>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

/**
 * Nodes (application steps)
 */
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

/**
 * Control flow
 */
import { StateGraph } from "@langchain/langgraph";

// Compile application and test
const graph = new StateGraph(StateAnnotation)
  .addNode("retrieve", retrieve)
  .addNode("generate", generate)
  .addEdge("__start__", "retrieve")
  .addEdge("retrieve", "generate")
  .addEdge("generate", "__end__")
  .compile();

// let inputs = { question: "Aonde será a inscrição?" };

// const result = await graph.invoke(inputs);
// console.log(result.context.slice(0, 2));
// console.log(`\nAnswer: ${result["answer"]}`);

/**
 * Steam steps
 */
// console.log(inputs);
// console.log("\n====\n");
// for await (const chunk of await graph.stream(inputs, {
//   streamMode: "updates",
// })) {
//   console.log(chunk);
//   console.log("\n====\n");
// }

/**
 * Stream tokens
 */
// let inputs = { question: "Como fazer inscrição no PID?" };


// const stream = await graph.stream(inputs, { streamMode: "messages" });

// for await (const [message, _metadata] of stream) {
//   process.stdout.write(message.content);
// }


// console.log({
//   totalDocuments, allSplits: allSplits[0]
// })

/**
 * Query analysis
 */
const totalDocuments = allSplits.length;
const third = Math.floor(totalDocuments / 3);

allSplits.forEach((document, i) => {
  if (i < third) {
    document.metadata["section"] = "beginning";
  } else if (i < 2 * third) {
    document.metadata["section"] = "middle";
  } else {
    document.metadata["section"] = "end";
  }
});


const vectorStoreQA = new MemoryVectorStore(embeddings);
await vectorStoreQA.addDocuments(allSplits);

import { z } from "zod";

const searchSchema = z.object({
  query: z.string().describe("Search query to run."),
  section: z.enum(["beginning", "middle", "end"]).describe("Section to query."),
});

const structuredLlm = llm.withStructuredOutput(searchSchema);

const StateAnnotationQA = Annotation.Root({
  question: Annotation<string>,
  search: Annotation<z.infer<typeof searchSchema>>,
  context: Annotation<Document[]>,
  answer: Annotation<string>,
});

const analyzeQuery = async (state: typeof InputStateAnnotation.State) => {
  const result = await structuredLlm.invoke(state.question);
  return { search: result };
};

const retrieveQA = async (state: typeof StateAnnotationQA.State) => {
  const filter = (doc) => doc.metadata.section === state.search.section;
  const retrievedDocs = await vectorStore.similaritySearch(
    state.search.query,
    2,
    filter
  );
  return { context: retrievedDocs };
};

const generateQA = async (state: typeof StateAnnotationQA.State) => {
  const docsContent = state.context.map((doc) => doc.pageContent).join("\n");
  const messages = await promptTemplate.invoke({
    question: state.question,
    context: docsContent,
  });
  const response = await llm.invoke(messages);
  return { answer: response.content };
};

const graphQA = new StateGraph(StateAnnotationQA)
  .addNode("analyzeQuery", analyzeQuery)
  .addNode("retrieveQA", retrieveQA)
  .addNode("generateQA", generateQA)
  .addEdge("__start__", "analyzeQuery")
  .addEdge("analyzeQuery", "retrieveQA")
  .addEdge("retrieveQA", "generateQA")
  .addEdge("generateQA", "__end__")
  .compile();
// allSplits[0].metadata;


let inputsQA = {
  question: "Aonde fazer inscrição?",
};

console.log(inputsQA);
console.log("\n====\n");
for await (const chunk of await graphQA.stream(inputsQA, {
  streamMode: "updates",
})) {
  console.log(chunk);
  console.log("\n====\n");
}
