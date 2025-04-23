// src/load.ts
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { PrismaClient } from "@prisma/client";
import fs from "fs/promises";
import path from "path";

const prisma = new PrismaClient();
const outputPath = path.resolve("vectorstore.json");

export async function getDataFromDB() {
  return await prisma.tb_processo_seletivo.findMany({
    select: {
      ds_titulo_processo_seletivo: true,
      tb_candidatura: {
        select: {
          fl_autodeclaracao_concorr_cota: true,
          ds_deficiencia: true,
          fl_portador_deficiencia_fisica: true,
        },
      },
    },
  });
}

export function serializeToText(obj: any): string {
  const candidaturas = obj.tb_candidatura.map((c: any, i: number) => {
    return `Candidatura ${i + 1}:
  - Autodeclarado cotista: ${c.fl_autodeclaracao_concorr_cota ? "Sim" : "Não"}
  - Deficiência: ${c.ds_deficiencia || "Nenhuma"}
  - Deficiência física: ${c.fl_portador_deficiencia_fisica ? "Sim" : "Não"}`;
  }).join("\n\n");

  return `Processo: ${obj.ds_titulo_processo_seletivo}\n\n${candidaturas}`;
}

export async function generateAndSaveEmbeddingsJSON() {
  const data = await getDataFromDB();
  const texts = data.map(serializeToText);

  const embedder = new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2",
  });

  const embeddings = await embedder.embedDocuments(texts);

  const jsonData = data.map((original, i) => ({
    metadata: original,
    text: texts[i],
    embedding: embeddings[i],
  }));

  await fs.writeFile(outputPath, JSON.stringify(jsonData, null, 2));
  console.log(`✅ Embeddings salvos no arquivo ${outputPath}`);
}

if (require.main === module) {
  generateAndSaveEmbeddingsJSON();
}
