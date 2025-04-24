import { OpenAIEmbeddings } from "@langchain/openai";
import "cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { z } from "zod";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatGroq } from "@langchain/groq";

import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";

const llm = new ChatGroq({
  model: "llama-3.3-70b-versatile",
  temperature: 0
});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});

const vectorStore = new MemoryVectorStore(embeddings);

// Load and chunk contents of the blog
const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  // "https://lilianweng.github.io/posts/2023-06-23-agent/",
  // 'https://www.prisma.io/blog/securely-access-prisma-postgres-from-the-frontend-early-access',
  // 'https://ge.globo.com/futebol/futebol-internacional/futebol-ingles/noticia/2025/04/18/casemiro-da-a-volta-por-cima-no-manchester-united-e-acende-debate-sobre-volta-a-selecao.ghtml',
  'https://defensoria.mg.def.br/defensoria-publica-faz-atendimento-presencial-e-remoto-para-pessoas-interessadas-em-aderir-ao-programa-de-indenizacao-referente-ao-rompimento-da-barragem-de-fundao/',
  {
    selector: pTagSelector,
  }
);

const docs = await cheerioLoader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const allSplits = await splitter.splitDocuments(docs);
// console.log(allSplits)
// Index chunks
await vectorStore.addDocuments(allSplits);

// const graph = new StateGraph(MessagesAnnotation);

const retrieveSchema = z.object({ query: z.string() });

const retrieve = tool(
  async ({ query }) => {
    const retrievedDocs = await vectorStore.similaritySearch(query, 2);
    const serialized = retrievedDocs
      .map(
        (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`
      )
      .join("\n");
    return [serialized, retrievedDocs];
  },
  {
    name: "retrieve",
    description: "Retrieve information related to a query.",
    schema: retrieveSchema,
    responseFormat: "content_and_artifact",
  }
);

// Step 1: Generate an AIMessage that may include a tool-call to be sent.
async function queryOrRespond(state: typeof MessagesAnnotation.State) {
  const llmWithTools = llm.bindTools([retrieve]);
  const response = await llmWithTools.invoke(state.messages);
  // MessagesState appends messages to state instead of overwriting
  return { messages: [response] };
}

// Step 2: Execute the retrieval.
const tools = new ToolNode([retrieve]);

// Step 3: Generate a response using the retrieved content.
async function generate(state: typeof MessagesAnnotation.State) {
  // Get generated ToolMessages
  let recentToolMessages = [];
  for (let i = state["messages"].length - 1; i >= 0; i--) {
    let message = state["messages"][i];
    if (message instanceof ToolMessage) {
      recentToolMessages.push(message);
    } else {
      break;
    }
  }
  let toolMessages = recentToolMessages.reverse();

  // Format into prompt
  const docsContent = toolMessages.map((doc) => doc.content).join("\n");
  const systemMessageContent =
    "You are an assistant for question-answering tasks. " +
    "Use the following pieces of retrieved context to answer " +
    "the question. If you don't know the answer, say that you " +
    "don't know. Use three sentences maximum and keep the " +
    "answer concise." +
    "\n\n" +
    `${docsContent}`;

  const conversationMessages = state.messages.filter(
    (message) =>
      message instanceof HumanMessage ||
      message instanceof SystemMessage ||
      (message instanceof AIMessage && message.tool_calls.length == 0)
  );
  const prompt = [
    new SystemMessage(systemMessageContent),
    ...conversationMessages,
  ];

  // Run
  const response = await llm.invoke(prompt);
  return { messages: [response] };
}

import { toolsCondition } from "@langchain/langgraph/prebuilt";

const graphBuilder = new StateGraph(MessagesAnnotation)
  .addNode("queryOrRespond", queryOrRespond)
  .addNode("tools", tools)
  .addNode("generate", generate)
  .addEdge("__start__", "queryOrRespond")
  .addConditionalEdges("queryOrRespond", toolsCondition, {
    __end__: "__end__",
    tools: "tools",
  })
  .addEdge("tools", "generate")
  .addEdge("generate", "__end__");

// const graph = graphBuilder.compile();

import { BaseMessage, isAIMessage } from "@langchain/core/messages";

const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message._getType()}]: ${message.content}`;
  if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};

// let inputs1 = { messages: [{ role: "user", content: "bom dia, tudo bem?" }] };

// for await (const step of await graph.stream(inputs1, {
//   streamMode: "values",
// })) {
//   const lastMessage = step.messages[step.messages.length - 1];
//   prettyPrint(lastMessage);
//   console.log("-----\n");
// }

// let inputs2 = {
//   messages: [{ role: "user", content: "Olá, como vai?" }],
// };

// for await (const step of await graph.stream(inputs2, {
//   streamMode: "values",
// })) {
//   const lastMessage = step.messages[step.messages.length - 1];
//   prettyPrint(lastMessage);
//   console.log("-----\n");
// }



import { MemorySaver } from "@langchain/langgraph";

const checkpointer = new MemorySaver();
const graphWithMemory = graphBuilder.compile({ checkpointer });

// Specify an ID for the thread
const threadConfig = {
  configurable: { thread_id: "abc123" },
  streamMode: "values" as const,
};


let inputs3 = {
  messages: [{ role: "user", content: "Quais os dias do multirão?" }],
};

for await (const step of await graphWithMemory.stream(inputs3, threadConfig)) {
  const lastMessage = step.messages[step.messages.length - 1];
  prettyPrint(lastMessage);
  console.log("-----\n");
}

// let inputs4 = {
//   messages: [
//     { role: "user", content: "E de quais times ele venceu?" },
//   ],
// };

// for await (const step of await graphWithMemory.stream(inputs4, threadConfig)) {
//   const lastMessage = step.messages[step.messages.length - 1];
//   prettyPrint(lastMessage);
//   console.log("-----\n");
// }

// import { createReactAgent } from "@langchain/langgraph/prebuilt";

// const agent = createReactAgent({ llm, tools: [retrieve], });

// async function main() {

//   let inputs5 = { messages: [{ role: 'user', content: 'Quais os dias do multirão?' }] };

//   console.time()
//   for await (const step of await agent.stream(inputs5, {
//     streamMode: "values",

//   })) {
//     const lastMessage = step.messages[step.messages.length - 1];
//     prettyPrint(lastMessage);
//     console.log("-----\n");
//   }

// let input2 = { messages: [{ role: 'user', content: 'Quais times ele venceu?' }] };

// const res2 = await agent.invoke(input2);
// console.log({ res2 });

// console.timeEnd();
// }
// main()
// Note: tslab only works inside a jupyter notebook. Don't worry about running this code yourself!
// import * as tslab from "tslab";
// 
// const image = await agent.getGraph().drawMermaidPng();
// const arrayBuffer = await image.arrayBuffer();

// await tslab.display.png(new Uint8Array(arrayBuffer));