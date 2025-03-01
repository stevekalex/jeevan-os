import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { z } from "zod";
import { tool } from "@langchain/core/tools";

// Changes to make
// 1. Embed vector store locally
// 2. Store vectors in your own DB project
// 3. Retrieve vectors from your DB project based on input
// 4. Use the vectors to generate a response

const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0,
  apiKey: process.env.OPENAI_API_KEY
});

const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-large"
});
const vectorStore = new MemoryVectorStore(embeddings);

const entries: Document[] = [
  {
    pageContent: "Today, I went to the park and saw a dog. Life is great at the moment. But another boring day at work, I wish I could do something else",
    metadata: {}
  }
]

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000, chunkOverlap: 200
});
const allSplits = await splitter.splitDocuments(entries);

// Index chunks
await vectorStore.addDocuments(allSplits);

// Define prompt for question-answering
const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

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


let inputs = { question: "Have I seen any animals recently?" };

export const getResult = async () => {
  const result = await graph.invoke(inputs);
  return result.answer
}

