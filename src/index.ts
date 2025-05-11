/**
 * Copyright 2024 Google LLC
 * Copyright 2024 LanceDB (modified for LanceDB)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  connect,
  Connection,
  Table,
} from '@lancedb/lancedb';
import { Genkit, z } from 'genkit';
import { GenkitPlugin, genkitPlugin } from 'genkit/plugin';
import { EmbedderArgument, Embedding } from 'genkit/embedder';
import {
  CommonRetrieverOptionsSchema,
  Document,
  indexerRef,
  retrieverRef,
} from 'genkit/retriever';
import { Md5 } from 'ts-md5';

export enum WriteMode {
  Create = 'create',
  Append = 'append', 
  Overwrite = 'overwrite'
}

const LanceDBRetrieverOptionsSchema = CommonRetrieverOptionsSchema.extend({
  k: z.number().int().positive().default(5),
  whereFilter: z.string().nullable().default(null),
  vectorColumnName: z.string().default('vector'),
  dbUri: z.string().optional().default(".db"),
  tableName: z.string().optional().default("table"),
});

const LanceDBIndexerOptionsSchema = z.object({
  dbUri: z.string().optional().default(".db"), 
  tableName: z.string().optional().default("table"),
  vectorColumnName: z.string().default('vector'),
  textColumnName: z.string().default('text'),
  metadataColumnName: z.string().default('metadata'), 
  writeMode: z.nativeEnum(WriteMode).default(WriteMode.Append),
});


interface LanceDbDataRow {
  id: string;
  vectorData?: number[];
  textData?: string;
  metadataJson?: string;
  [key: string]: any;
}


/**
 * lancedbRetrieverRef function creates a retriever reference for LanceDB.
 * @param params The params for the new LanceDB retriever
 * @param params.tableName The tableName for the LanceDB retriever
 * @param params.displayName A display name for the retriever. If not specified, defaults to `LanceDB - <tableName>`
 * @returns A reference to a LanceDB retriever.
 */
export const lancedbRetrieverRef = (params: {
  tableName?: string;
  displayName?: string;
}) => {
  const effectiveTableName = params.tableName || "table";
  return retrieverRef({
    name: `lancedb/${effectiveTableName}`,
    info: {
      label: params.displayName ?? `LanceDB - ${effectiveTableName}`,
    },
    configSchema: LanceDBRetrieverOptionsSchema,
  });
};

/**
 * lancedbIndexerRef function creates an indexer reference for LanceDB.
 * @param params The params for the new LanceDB indexer.
 * @param params.tableName The tableName for the LanceDB indexer.
 * @param params.displayName A display name for the indexer. If not specified, defaults to `LanceDB - <tableName>`
 * @returns A reference to a LanceDB indexer.
 */
export const lancedbIndexerRef = (params: {
  tableName?: string;
  displayName?: string;
}) => {
  const effectiveTableName = params.tableName || "table";
  return indexerRef({
    name: `lancedb/${effectiveTableName}`,
    info: {
      label: params.displayName ?? `LanceDB - ${effectiveTableName}`,
    },
    configSchema: LanceDBIndexerOptionsSchema,
  });
};

// --- Plugin Definition ---

/**
 * LanceDB plugin that provides a LanceDB retriever and indexer.
 * @param configs An array of configurations, each containing:
 * dbUri: The LanceDB connection URI (e.g., "./.lancedb").
 * tableName: The name of the table for this indexer/retriever.
 * embedder: The embedder to use.
 * embedderOptions: Optional options for the embedder.
 * vectorColumnName: (Optional) Name of the vector column (default: "vector").
 * textColumnName: (Optional) Name of the text content column (default: "text").
 * metadataColumnName: (Optional) Name of the metadata column (default: "metadata").
 * @returns The LanceDB Genkit plugin.
 */
export function lancedb<EmbedderCustomOptions extends z.ZodTypeAny>(
  configs: {
    dbUri?: string;
    tableName?: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    vectorColumnName?: string;
    textColumnName?: string;
    metadataColumnName?: string;
  }[]
): GenkitPlugin {
  return genkitPlugin('lancedb', async (ai: Genkit) => {
    configs.forEach((config) => {
      const effectiveConfig = {
        ...config,
        dbUri: config.dbUri || ".db",
        tableName: config.tableName || "table"
      };
      configureLanceDBRetriever(ai, effectiveConfig);
      configureLanceDBIndexer(ai, effectiveConfig);
    });
  });
}

export default lancedb; // Default export for convenience

// --- Retriever Configuration ---

export function configureLanceDBRetriever<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    dbUri?: string;
    tableName?: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    vectorColumnName?: string;
    textColumnName?: string;
    metadataColumnName?: string;
  }
) {
  const {
    dbUri = ".db",
    tableName = "table",
    embedder,
    embedderOptions,
    textColumnName = 'text',
    metadataColumnName = 'metadata',
  } = params;

  // Define the retriever
  ai.defineRetriever(
    {
      name: `lancedb/${tableName}`,
      configSchema: LanceDBRetrieverOptionsSchema,
    },
    async (
      content: Document,
      options: z.infer<typeof LanceDBRetrieverOptionsSchema>
    ): Promise<{ documents: Document[] }> => {
      let db: Connection | null = null;
      let tbl: Table | null = null;
      try {
        db = await connect(dbUri);
        tbl = await db.openTable(tableName);
      } catch (error: any) {
        console.error(
          `LanceDB Retriever Error: Failed to connect or open table '${tableName}' at '${dbUri}'. Does it exist?`,
          error
        );
        return { documents: [] };
      }


      const queryEmbeddings: Embedding[] = await ai.embed({
        embedder,
        content: content,
        options: embedderOptions,
      });

      if (!queryEmbeddings || queryEmbeddings.length === 0) {
        console.warn('LanceDB Retriever: Query embedding resulted in no vectors.');
        return { documents: [] };
      }
      const queryVector = queryEmbeddings[0].embedding; 

      let searchQuery = tbl
        .search(queryVector) 
        .limit(options.k);

      if (options.whereFilter) {
        searchQuery = searchQuery.where(options.whereFilter);
      }

      const selectColumns = Array.from(
        new Set([textColumnName, metadataColumnName])
      ); 
      searchQuery = searchQuery.select(selectColumns);

      try {
        const results: Record<string, any>[] = await searchQuery.toArray(); // Execute query

        const documents: Document[] = results.map((res) => {
          const docContent = res[textColumnName] ?? '';
          const metadataStr = res[metadataColumnName] ?? '{}';
          let docMetadata: Record<string, unknown> = {};

          try {
            docMetadata = JSON.parse(metadataStr);
          } catch (e) {
            console.warn(
              `LanceDB Retriever: Failed to parse metadata for a result from table '${tableName}'. Content: "${metadataStr}"`,
              e
            );
            docMetadata = { error: 'Failed to parse metadata' };
          }

          return Document.fromData(docContent, undefined, docMetadata);
        });

        return { documents };
      } catch (searchError: any) {
        console.error(
          `LanceDB Retriever Error: Search failed for table '${tableName}'`,
          searchError
        );
        return { documents: [] }; 
      }
    }
  );
}


export function configureLanceDBIndexer<
  EmbedderCustomOptions extends z.ZodTypeAny,
>(
  ai: Genkit,
  params: {
    dbUri?: string;
    tableName?: string;
    embedder: EmbedderArgument<EmbedderCustomOptions>;
    embedderOptions?: z.infer<EmbedderCustomOptions>;
    vectorColumnName?: string;
    textColumnName?: string;
    metadataColumnName?: string;
  }
) {
  const {
    dbUri = ".db",
    tableName = "table",
    embedder,
    embedderOptions,
    vectorColumnName = 'vector',
    textColumnName = 'text',
    metadataColumnName = 'metadata',
  } = params;

  ai.defineIndexer(
    {
      name: `lancedb/${tableName}`,
      configSchema: LanceDBIndexerOptionsSchema,
    },
    async (
      docs: Document[],
      options?: z.infer<typeof LanceDBIndexerOptionsSchema> 
    ) => {
      const effectiveDbUri = options?.dbUri ?? dbUri;
      const effectiveTableName = options?.tableName ?? tableName;
      const effectiveVectorCol = options?.vectorColumnName ?? vectorColumnName;
      const effectiveTextCol = options?.textColumnName ?? textColumnName;
      const effectiveMetadataCol = options?.metadataColumnName ?? metadataColumnName;
      const effectiveWriteMode = options?.writeMode ?? WriteMode.Append;

      if (!docs || docs.length === 0) {
        console.log('LanceDB Indexer: No documents provided to index.');
        return;
      }

      let db: Connection;
      try {
        db = await connect(effectiveDbUri);
      } catch (error: any) {
        console.error(
          `LanceDB Indexer Error: Failed to connect to '${effectiveDbUri}'`,
          error
        );
        throw new Error(
          `LanceDB Indexer: Connection failed to ${effectiveDbUri}`
        );
      }

      let tableExists = true;
      let tbl: Table | null = null;
      try {
        tbl = await db.openTable(effectiveTableName);
        console.log(
          `LanceDB Indexer: Using existing table '${effectiveTableName}'. Mode: ${effectiveWriteMode}`
        );
        if (effectiveWriteMode === WriteMode.Overwrite) {
          console.warn(
            `LanceDB Indexer: Overwriting existing table '${effectiveTableName}'!`
          );
        }
      } catch (e) {
        tableExists = false;
        console.log(
          `LanceDB Indexer: Table '${effectiveTableName}' not found. Will attempt to create.`
        );
      }

      const embeddingPromises = docs.map((doc) =>
        ai.embed({
          embedder,
          content: doc,
          options: embedderOptions,
        })
      );
      const embeddingsForEachDoc: Embedding[][] = await Promise.all(
        embeddingPromises
      );

      const dataToAdd: LanceDbDataRow[] = [];
      docs.forEach((doc, i) => {
        const docEmbeddings: Embedding[] = embeddingsForEachDoc[i];

        docEmbeddings.forEach((embedding) => {
          const docRepr = `${doc.data || doc.toString()}-${JSON.stringify(doc.metadata || {})}`;
          const id = Md5.hashStr(docRepr);

          let metadataJson = '{}';
          try {
            metadataJson = JSON.stringify(doc.metadata || {});
          } catch (jsonError) {
            console.warn(
              `LanceDB Indexer: Could not stringify metadata for doc ${id}`,
              jsonError
            );
          }

          const row: LanceDbDataRow = {
            id: id,
            [effectiveVectorCol]: embedding.embedding,
            [effectiveTextCol]: doc.data || doc.toString(),
            [effectiveMetadataCol]: metadataJson,
          };
          dataToAdd.push(row);
        });
      });

      if (dataToAdd.length === 0) {
        console.log('LanceDB Indexer: No data rows generated after embedding.');
        return;
      }

      try {
        if (!tableExists) {
          console.log(
            `LanceDB Indexer: Creating table '${effectiveTableName}' with ${dataToAdd.length} records.`
          );
          tbl = await db.createTable(effectiveTableName, dataToAdd, {
            mode: WriteMode.Create,
          });
          console.log(
            `LanceDB Indexer: Successfully created table '${effectiveTableName}'.`
          );
        } else if (tbl) {
          if (effectiveWriteMode === WriteMode.Overwrite) {
            console.log(
              `LanceDB Indexer: Overwriting table '${effectiveTableName}' with ${dataToAdd.length} records.`
            );
            tbl = await db.createTable(effectiveTableName, dataToAdd, {
              mode: WriteMode.Overwrite,
            });
          } else {
            console.log(
              `LanceDB Indexer: Appending ${dataToAdd.length} records to table '${effectiveTableName}'.`
            );
            await tbl.add(dataToAdd);
          }
          console.log(
            `LanceDB Indexer: Successfully added data to table '${effectiveTableName}'.`
          );
        }
      } catch (writeError: any) {
        console.error(
          `LanceDB Indexer Error: Failed to write data to table '${effectiveTableName}'`,
          writeError
        );
        throw new Error(
          `LanceDB Indexer: Failed writing to table ${effectiveTableName}`
        );
      }
    }
  );
}

// --- Optional Helper Functions ---

/**
 * Helper function for creating a LanceDB table explicitly.
 * Primarily useful if you need to create an empty table or ensure specific settings.
 * @param params Parameters for table creation
 * @param params.dbUri Connection URI
 * @param params.tableName Name of the table to create
 * @param params.exampleData Optional example data (list of objects) to infer schema.
 * LanceDB TS SDK primarily uses data inference. Defining schema
 * programmatically might require Arrow schema objects.
 */
export async function createLanceDBTable(params: {
  dbUri: string;
  tableName: string;
  exampleData?: Record<string, any>[];
}): Promise<Table> {
  const { dbUri, tableName, exampleData } = params;
  if (!exampleData || exampleData.length === 0) {
    throw new Error(
      'LanceDB Helper: Example data must be provided to infer schema for table creation in TS SDK.'
    );
  }
  const db = await connect(dbUri);
  console.log(`LanceDB Helper: Creating table '${tableName}' at '${dbUri}'...`);
  const tbl = await db.createTable(tableName, exampleData, {
    mode: WriteMode.Create,
  });
  console.log(`LanceDB Helper: Table '${tableName}' created successfully.`);
  return tbl;
}

/**
 * Helper function for deleting a LanceDB table.
 * @param params Parameters for deletion
 * @param params.dbUri Connection URI
 * @param params.tableName Name of the table to delete
 */
export async function deleteLanceDBTable(params: {
  dbUri: string;
  tableName: string;
}): Promise<void> {
  const { dbUri, tableName } = params;
  const db = await connect(dbUri);
  console.log(`LanceDB Helper: Deleting table '${tableName}' from '${dbUri}'...`);
  await db.dropTable(tableName);
  console.log(`LanceDB Helper: Table '${tableName}' deleted successfully.`);
}


/**
 * Helper function to define both an indexer and retriever for LanceDB in one call.
 * @param params Configuration parameters
 * @param params.tableName The name of the table to use
 * @param params.displayName Optional display name for the components
 * @returns An object containing references to both the indexer and retriever
 */
export function defineLanceDBComponents(params: {
  tableName?: string;
  displayName?: string;
}): {
  indexer: ReturnType<typeof lancedbIndexerRef>;
  retriever: ReturnType<typeof lancedbRetrieverRef>;
} {
  const effectiveTableName = params.tableName || "table";
  const effectiveDisplayName = params.displayName || `LanceDB - ${effectiveTableName}`;
  
  return {
    indexer: lancedbIndexerRef({
      tableName: effectiveTableName,
      displayName: effectiveDisplayName,
    }),
    retriever: lancedbRetrieverRef({
      tableName: effectiveTableName,
      displayName: effectiveDisplayName,
    }),
  };
}