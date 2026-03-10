Architecture
============

This section describes the high-level architecture of HYPPO and how its components
interact during feature extraction.


System Overview
---------------

.. mermaid::

   flowchart LR
       subgraph Input
           H5[".h5 File"]
           YAML["Config YAML/JSON"]
       end

       subgraph Core
           HSI["HSI"]
           FS["FeatureSpace"]
           FG["FeatureGraph"]
       end

       subgraph Execution
           Runner["Runner"]
           SEQ["SequentialRunner"]
           DT["DaskThreadsRunner"]
           DP["DaskProcessesRunner"]
       end

       subgraph Output
           FC["FeatureCollection"]
           Feature["Feature"]
       end

       H5 -->|"io.load_h5_hsi()"| HSI
       YAML -->|"io.load_config_yaml()"| FS
       FS --> FG
       HSI --> Runner
       FG --> Runner
       Runner --> FC
       FC --> Feature

       Runner -.-> SEQ
       Runner -.-> DT
       Runner -.-> DP


Extraction Pipeline
-------------------

The following diagram shows the execution flow when ``FeatureSpace.extract()`` is called:

.. mermaid::

   sequenceDiagram
       participant User
       participant FS as FeatureSpace
       participant FG as FeatureGraph
       participant Runner
       participant Ext as Extractor

       User->>FS: extract(hsi, runner)
       FS->>FG: get_execution_order()
       FG-->>FS: topological order
       FS->>Runner: resolve(hsi, feature_space)

       loop For each extractor in order
           Runner->>Runner: build input kwargs
           Runner->>Ext: extract(hsi, **inputs)
           Ext->>Ext: _validate(hsi, **inputs)
           Ext->>Ext: _extract(hsi, **inputs)
           Ext-->>Runner: result dict
       end

       Runner-->>FS: FeatureCollection
       FS-->>User: FeatureCollection


Dependency Resolution
---------------------

Extractors can declare dependencies on other extractors. HYPPO builds a directed
acyclic graph (DAG) and resolves execution order via topological sorting.

.. mermaid::

   graph TD
       Mean["MeanExtractor"]
       Std["StdExtractor"]
       PCA["PCAExtractor"]
       ICA["ICAExtractor"]
       NDVI["NDVIExtractor"]
       GLCM["GLCMExtractor"]
       MP["MPExtractor"]

       PCA -->|"pca_features"| ICA
       Mean --> MP
       Std --> MP

       style Mean fill:#4a9eff,color:#fff
       style Std fill:#4a9eff,color:#fff
       style PCA fill:#4a9eff,color:#fff
       style NDVI fill:#4a9eff,color:#fff
       style GLCM fill:#66bb6a,color:#fff
       style ICA fill:#ffa726,color:#fff
       style MP fill:#ffa726,color:#fff

   **Legend:**

   - |blue| Independent extractors (no dependencies)
   - |green| Spatial extractors
   - |orange| Extractors with dependencies

.. |blue| raw:: html

   <span style="color:#4a9eff">&#9632;</span>

.. |green| raw:: html

   <span style="color:#66bb6a">&#9632;</span>

.. |orange| raw:: html

   <span style="color:#ffa726">&#9632;</span>

Extractors without dependencies can run in parallel when using a Dask runner.
Dependent extractors wait for their inputs to be computed first.


Extractor Class Hierarchy
-------------------------

.. mermaid::

   classDiagram
       class Extractor {
           <<abstract>>
           +extract(data, **inputs) dict
           +_validate(data, **inputs)
           +_extract(data, **inputs)* dict
           +feature_name()$ str
           +get_input_dependencies()$ dict
           +get_input_default(input_name)$ Extractor
       }

       Extractor <|-- NDVIExtractor
       Extractor <|-- NDWIExtractor
       Extractor <|-- SAVIExtractor
       Extractor <|-- PCAExtractor
       Extractor <|-- ICAExtractor
       Extractor <|-- MNFExtractor
       Extractor <|-- GLCMExtractor
       Extractor <|-- LBPExtractor
       Extractor <|-- GaborExtractor
       Extractor <|-- DWT1DExtractor
       Extractor <|-- DWT2DExtractor
       Extractor <|-- DWT3DExtractor
       Extractor <|-- MPExtractor
       Extractor <|-- PPExtractor
       Extractor <|-- GeometricMomentExtractor
       Extractor <|-- LegendreMomentExtractor
       Extractor <|-- ZernikeMomentExtractor


Runner Class Hierarchy
----------------------

.. mermaid::

   classDiagram
       class BaseRunner {
           <<abstract>>
           +resolve(data, feature_space)* FeatureCollection
           +_get_defaults_for_extractor(extractor) dict
       }

       class DaskRunner {
           -_client: Client
           -_cluster: Cluster
           +resolve(data, feature_space) FeatureCollection
           -_build_dask_graph(data, feature_graph) dict
       }

       BaseRunner <|-- SequentialRunner
       BaseRunner <|-- LocalProcessRunner
       BaseRunner <|-- DaskRunner
       DaskRunner <|-- DaskThreadsRunner
       DaskRunner <|-- DaskProcessesRunner


Registry Pattern
----------------

Both extractors and runners use a singleton registry pattern for dynamic
lookup by name:

.. mermaid::

   flowchart LR
       subgraph ExtractorRegistry
           ER["ExtractorRegistry (singleton)"]
           ER --- E1["'NDVIExtractor' → NDVIExtractor"]
           ER --- E2["'PCAExtractor' → PCAExtractor"]
           ER --- E3["'GLCMExtractor' → GLCMExtractor"]
           ER --- E4["..."]
       end

       subgraph RunnerRegistry
           RR["RunnerRegistry (singleton)"]
           RR --- R1["'sequential' → SequentialRunner"]
           RR --- R2["'dask-threads' → DaskThreadsRunner"]
           RR --- R3["'dask-processes' → DaskProcessesRunner"]
       end

The registries are used by the configuration loader to instantiate extractors
and runners from their string names in YAML/JSON configuration files.
