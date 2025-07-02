# Multi-Lingual RAG System

This project is a simple demonstration of a multi-lingual Retrieval-Augmented Generation (RAG)

## RAG system

I use two vector fields: `title` and `description_text` column values (see the
[data](#data) section). During the preprocessing step, I split the latter
values into smaller chunks to improve the retrieval accuracy. The folder IDs in
the metadata help to link the chunks to the whole documents to be used in the
LLM context.

Japanese texts require special treatment since they are not split by spaces,
use different punctuation chars, and encoding requires more bytes per symbol.
On average, one hieroglyph carries more information than a Latin character, so
I use a different chunk size for Japanese texts, and also this is because of
limits on string length being set per text column in the MilvusDB.

## Evaluation Pipeline

I create small equal subsets of data for each language. Though small, they are
enough to build Knowledge Graphs (KG) for further generation of questions and
ground truth answers over the KG. The prompts for the data synthesis and
LLM scorer's prompts are adapted to each language.

## Data

The system works with a multi-lingual knowledge base (comes as a CSV file)
containing around 6500 samples distributed across four languages. There are
three GPT-generated scripts in `scripts` folder to inspect the data.

### Language Distribution

- English (en): 32.7% (2120 samples)
- Japanese (ja): 29.3% (1902 samples)
- Hebrew (he_IL): 22.3% (1444 samples)
- Russian (ru): 15.7% (1017 samples)

### Data Structure

The knowledge base contains the following information:

- **id**: Document identifier
- **lang**: Language code (en, ja, he_IL, ru)
- **categoryId**: Question category (23 unique categories)
- **folderId**: Organizational folder
- **title**: Question text
- **description**: HTML answer content
- **description_text**: Plain text version of the answer

### Content Characteristics

- Average content length: ~1037 characters
- Content size distribution:
  - ~2000 samples > 1000 characters
  - ~30 samples > 10000 characters
  - Largest sample: ~30000 characters

### Cross-Language Categories

The knowledge base maintains consistent categorization across languages with 20
universal categories present in all languages, providing a coherent
multi-lingual knowledge structure.

## Technical Stack

### RAG Pipeline

The RAG implementation leverages the following components:

- **litellm**: For LLMs integration and inference
- **Milvus**: Vector database for storing and retrieving embeddings

### Evaluation Framework

The evaluation pipeline utilizes:

- **Ragas**: For RAG-specific evaluation metrics
- **langchain**: Since it is well-integrated into Ragas

## Configuration

The system provides comprehensive configuration options through YAML files:

- **rag-config.yaml**: Main configuration for the RAG pipeline

  - Input data path and preprocessing settings
  - Database connection and collection parameters
  - Retrieval strategy and ranking weights
  - LLM model selection and parameters

- **test-config.yaml**: Configuration for evaluation and synthetic data generation
  - Test dataset sizes and distribution
  - Evaluation LLM selection
  - Synthetic data parameters

## Installation

### Prerequisites

- **Docker**: Docker installation with the `compose` subcommand
- **uv**: Python package and project manager

### Setup

1. Run the bootstrap script to set up the environment:

```bash
./bootstrap.sh
```

This script:

- Fetches the docker compose file for Milvus standalone deployment
- Sets up your OpenAI API key in the proper environment variable

## Usage

To run the RAG system from CLI:

```bash
uv run python src/main.py "your question goes here"
```

To evaluate the RAG system:

```bash
uv run --extra eval python src/eval/main.py
```

To run the jupyter notebook

```bash
uv run --extra notebook jupyter notebook
```
