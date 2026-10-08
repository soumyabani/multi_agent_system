# Sample Incident Data

This directory contains sample incident and knowledge base documents for the multi-agent system.

## Files

- `sample_incidents.json` - 10 realistic incident reports with root causes, impact analysis, and prevention measures

## Data Format

Each incident document contains:

- `id` - Unique incident identifier (e.g., INC-001)
- `title` - Short incident summary
- `content` - Detailed incident report including root cause, fix, and lessons learned
- `severity` - Incident severity: critical, high, medium, low
- `status` - Incident status: resolved, ongoing, investigating
- `created_at` - ISO 8601 timestamp when incident was created
- `resolved_at` - ISO 8601 timestamp when incident was resolved
- `root_cause` - One-line summary of the root cause
- `components_affected` - List of affected system components
- `tags` - Searchable tags for categorization (e.g., database, performance, deployment)

## Incident Categories

The sample data covers common production issues:

1. **Database Performance** - Connection pool exhaustion, replication lag, indexing
2. **Infrastructure** - Load balancer config, certificate expiration, disk space
3. **Deployment** - Backward incompatibility, rollback procedures
4. **Microservices** - Network partitions, inter-service communication
5. **External Dependencies** - Third-party API rate limits, external connectivity
6. **Monitoring** - Memory leaks, performance degradation

## Using This Data

To ingest these incidents into Qdrant, use the ingestion script:

```bash
python src/tools/ingest_incidents.py data/sample_incidents.json
```

This will:
1. Read each incident from the JSON file
2. Embed the content using the configured embedding model
3. Store in Qdrant with metadata for retrieval

After ingestion, you can ask the system incident-related questions like:

```bash
python main.py "What causes payment processing delays?"
python main.py "How do we handle database performance issues?"
python main.py "What's the process for deployment rollbacks?"
```

## Adding More Data

To add more incidents:

1. Add new objects to `sample_incidents.json` following the same format
2. Ensure each has a unique `id`
3. Provide detailed `content` for better retrieval and analysis
4. Re-run the ingestion script to update Qdrant
