use google_cloud_aiplatform_v1::client::PredictionService;
use google_cloud_gax::options::ClientConfig;

/// The model reference and number of dimensions
pub struct Model<'a> {
    pub publisher: &'a str,
    pub name: &'a str,
    pub ndims: usize,
}

impl<'a> Model<'a> {
    const fn new_google(name: &'a str) -> Self {
        Self {
            publisher: "google",
            name,
            ndims: 768,
        }
    }
}

/// Google's `text-embedding-045` embedding model
pub const TEXT_EMBEDDING_004: Model = Model::new_google("text-embedding-004");
/// Google's `text-embedding-005` embedding model
pub const TEXT_EMBEDDING_005: Model = Model::new_google("text-embedding-005");
/// Google's `text-multilingual-embedding-002` multilingual embedding model
pub const TEXT_MULTILINGUAL_EMBEDDING_002: Model =
    Model::new_google("text-multilingual-embedding-002");

// Convert the VertexAI error into a generic EmbeddingError::ProviderError
fn into_provider_error<E: std::error::Error>(error: E) -> rig::embeddings::EmbeddingError {
    rig::embeddings::EmbeddingError::ProviderError(error.to_string())
}

#[derive(Clone)]
pub struct EmbeddingModel {
    client: PredictionService,
    endpoint: String,
    ndims: usize,
    task_type: Option<api_types::TaskType>,
}

impl EmbeddingModel {
    pub async fn new(
        project: &str,
        location: &str,
        model: &Model<'_>,
        task_type: Option<api_types::TaskType>,
    ) -> Result<Self, rig::embeddings::EmbeddingError> {
        Ok(Self {
            client: PredictionService::new_with_config(ClientConfig::new().enable_tracing())
                .await
                .map_err(into_provider_error)?,
            endpoint: format!(
                "projects/{project}/locations/{location}/publishers/{}/models/{}",
                model.publisher, model.name,
            ),
            ndims: model.ndims,
            task_type,
        })
    }
}

impl rig::embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 1000;

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<rig::embeddings::Embedding>, rig::embeddings::EmbeddingError> {
        // NOTE: we're forced to do this because Embedding require returning the input
        let documents = texts.into_iter().collect::<Vec<_>>();

        let response = self
            .client
            .predict(self.endpoint.clone())
            .set_instances(documents.iter().map(|content| {
                serde_json::to_value(api_types::Instance {
                    content,
                    task_type: self.task_type,
                    title: None,
                })
                .unwrap()
            }))
            .send()
            .await
            .map_err(into_provider_error)?;

        let predictions: Vec<api_types::Prediction> =
            serde_json::from_value(serde_json::Value::Array(response.predictions))?;

        Ok(predictions
            .into_iter()
            .zip(documents)
            .map(|(prediction, document)| rig::embeddings::Embedding {
                document,
                vec: prediction.embeddings.values,
            })
            .collect())
    }
}

mod api_types {
    use serde::{Deserialize, Serialize};

    /// https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#tasktype
    #[derive(Clone, Copy, Serialize)]
    #[serde[rename_all="SCREAMING_SNAKE_CASE"]]
    pub enum TaskType {
        /// Specifies the given text is a query in a search or retrieval setting.
        RetrievalQuery,
        /// Specifies the given text is a document in a search or retrieval setting.
        RetrievalDocument,
        /// Specifies the given text is used for Semantic Textual Similarity (STS).
        SemanticSimilarity,
        /// Specifies that the embedding is used for classification.
        Classification,
        /// Specifies that the embedding is used for clustering.
        Clustering,
        /// Specifies that the query embedding is used for answering questions. Use RETRIEVAL_DOCUMENT for the document side.
        QuestionAnswering,
        /// Specifies that the query embedding is used for fact verification.
        FactVerification,
        /// Specifies that the query embedding is used for code retrieval for Java and Python.
        CodeRetrievalQuery,
    }

    #[derive(Serialize)]
    pub(super) struct Instance<'a> {
        /// The text that you want to generate embeddings for.
        pub(super) content: &'a str,
        /// Used to convey intended downstream application to help the model produce better embeddings. If left blank, the default used is RETRIEVAL_QUERY.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub(super) task_type: Option<TaskType>,
        /// Used to help the model produce better embeddings. Only valid with task_type=RETRIEVAL_DOCUMENT.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub(super) title: Option<&'a str>,
    }

    #[derive(Serialize)]
    #[allow(unused)]
    pub(super) struct Parameters {
        /// When set to true, input text will be truncated. When set to false, an error is returned if the input text is longer than the maximum length supported by the model. Defaults to true.
        #[serde(skip_serializing_if = "Option::is_none")]
        auto_truncate: Option<bool>,
        /// Used to specify output embedding size. If set, output embeddings will be truncated to the size specified.
        #[serde(skip_serializing_if = "Option::is_none")]
        output_dimensionality: Option<usize>,
    }

    #[derive(Deserialize)]
    #[allow(unused)]
    pub(super) struct Statistics {
        /// Indicates if the input text was longer than max allowed tokens and truncated.
        pub(super) truncated: bool,
        /// Number of tokens of the input text.
        pub(super) token_count: usize,
    }

    #[derive(Deserialize)]
    #[allow(unused)]
    pub(super) struct Embeddings {
        /// The statistics computed from the input text.
        pub(super) statistics: Statistics,
        /// The values field contains the embedding vectors corresponding to the words in the input text.
        pub(super) values: Vec<f64>,
    }

    #[derive(Deserialize)]
    pub(super) struct Prediction {
        /// The result generated from input text.
        pub(super) embeddings: Embeddings,
    }
}

#[cfg(test)]
mod test {
    use rig::embeddings::EmbeddingModel;

    async fn test_client() -> super::EmbeddingModel {
        super::EmbeddingModel::new(
            &std::env::var("GOOGLE_CLOUD_PROJECT").unwrap(),
            "us-central1",
            &super::TEXT_EMBEDDING_005,
            None,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_one() {
        let client = test_client().await;
        let embedding = client.embed_text("foo bar").await.unwrap();
        assert_eq!(embedding.vec.len(), 768);
    }

    #[tokio::test]
    async fn test_many() {
        let client = test_client().await;
        let embeddings = client
            .embed_texts(["foo bar".to_string(), "qux baz".to_string()])
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
