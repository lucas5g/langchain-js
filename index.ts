import { Prisma, PrismaClient, tb_processo_seletivo } from "@prisma/client";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import fs from "fs/promises";

interface ProcessInterface extends Prisma.tb_processo_seletivoGetPayload<{ include: { tb_candidatura: true } }> { }

const prisma = new PrismaClient();

async function getDataPrisma() {
  return await prisma.tb_processo_seletivo.findMany({
    select: {
      co_seq_processo_seletivo: true,
      ds_titulo_processo_seletivo: true,
      tb_candidatura: {
        select: {
          nu_inscricao_processo_seletivo: true,
          fl_autodeclaracao_concorr_cota: true,
          ds_deficiencia: true,
          fl_portador_deficiencia_fisica: true
        }
      }
    }
  });
}

function serializeToText(process: Partial<ProcessInterface>) {
  const candidaturas = process.tb_candidatura?.map(row => {
    return `Candidatura ${row.nu_inscricao_processo_seletivo}:
  - Autodeclarado cotista: ${row.fl_autodeclaracao_concorr_cota ? "Sim" : "Não"}
  - Deficiência: ${row.ds_deficiencia || "Nenhuma"}
  - Deficiência física: ${row.fl_portador_deficiencia_fisica ? "Sim" : "Não"}`;
  }).join("\n\n");

  return `Processo: ${process.ds_titulo_processo_seletivo}\n\n${candidaturas}`;
}

async function embedder(texts: string[]) {
  const model = new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2",


  });

  return await model.embedDocuments(texts);
}

async function indexDocuments(
  docs: Partial<ProcessInterface>[],
  storePath = "./embeddings.json"
) {
  const texts = docs.map(serializeToText);
  const embeddings = await embedder(texts);

  const dataToSave = docs.map((doc, i) => ({
    id: doc.co_seq_processo_seletivo?.toString(),
    text: texts[i],
    embedding: embeddings[i],
  }));

  await fs.writeFile(storePath, JSON.stringify(dataToSave, null, 2));
  console.log("✔ Embeddings salvos em JSON!");
}

// Execução principal
// const data = await getDataPrisma();
// await indexDocuments(data);



// Recarregar os dados com embeddings salvos
async function loadEmbeddingsFromJSON() {
  const data = await fs.readFile('./embeddings.json', "utf-8");
  return JSON.parse(data);
}

// Criar o vetor da query
async function embedQuery(text: string) {
  const model = new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2",
  });
  return await model.embedQuery(text);
}

// Calcular similaridade (cosine similarity)
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dot = vecA.reduce((acc, val, i) => acc + val * vecB[i], 0);
  const normA = Math.sqrt(vecA.reduce((acc, val) => acc + val * val, 0));
  const normB = Math.sqrt(vecB.reduce((acc, val) => acc + val * val, 0));
  return dot / (normA * normB);
}

// Buscar os documentos mais parecidos
async function search(query: string, storePath: string, topK = 3) {
  const docs = await loadEmbeddingsFromJSON();
  const queryEmbedding = await embedQuery(query);

  const scored = docs.map(doc => ({
    ...doc,
    score: cosineSimilarity(queryEmbedding, doc.embedding),
  }));

  return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

// Exemplo de uso
const resultados = await search("tem alguma vaga para pessoas com deficiência?", "path/para/seu/store.json");
console.log(resultados.map(r => r.text));

