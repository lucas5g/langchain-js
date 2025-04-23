import { ChatGroq } from "@langchain/groq";
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";

const datasource = new DataSource({
  type: "sqlite",
  database: "Chinook.db",
});
const db = await SqlDatabase.fromDataSourceParams({
  appDataSource: datasource,
});

const res = await db.run("SELECT * FROM Artist LIMIT 10;");
// console.log(JSON.parse(res));

const llm = new ChatGroq({
  model: 'meta-llama/llama-4-maverick-17b-128e-instruct',
  apiKey: process.env.GROQ_API_KEY
})


import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const queryPromptTemplate = await pull<ChatPromptTemplate>(
  "langchain-ai/sql-query-system-prompt"
);

queryPromptTemplate.promptMessages.forEach((message) => {
  // console.log(message.lc_kwargs.prompt.template);
});

import { z } from "zod";

const queryOutput = z.object({
  query: z.string().describe("Syntactically valid SQL query."),
});

const structuredLlm = llm.withStructuredOutput(queryOutput);

const writeQuery = async (state: typeof InputStateAnnotation.State) => {
  const promptValue = await queryPromptTemplate.invoke({
    dialect: db.appDataSourceOptions.type,
    top_k: 10,
    table_info: await db.getTableInfo(),
    input: state.question,
  });
  const result = await structuredLlm.invoke(promptValue);
  return { query: result.query };
};


const res2 = await writeQuery({ question: "How many Employees are there?" });
console.log(res2)

import { QuerySqlTool } from "langchain/tools/sql";

const executeQuery = async (state: typeof StateAnnotation.State) => {
  const executeQueryTool = new QuerySqlTool(db);
  return { result: await executeQueryTool.invoke(state.query) };
};

const generateAnswer = async (state: typeof StateAnnotation.State) => {
  const promptValue =
    "Given the following user question, corresponding SQL query, " +
    "and SQL result, answer the user question.\n\n" +
    `Question: ${state.question}\n` +
    `SQL Query: ${state.query}\n` +
    `SQL Result: ${state.result}\n`;
  const response = await llm.invoke(promptValue);
  return { answer: response.content };
};

const res3 = await generateAnswer()