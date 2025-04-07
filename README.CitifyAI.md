# Agno in CitifyAI

This is a clone of the [Agno](https://github.com/agno-agi/agno) repository, which provides agent routing and multi-step planning capabilities for our CitifyAI project.

## Integration Strategy

Agno is a Python-based framework, while CitifyAI is built with Next.js. To integrate these technologies:

1. We use Next.js API routes as a bridge between our TypeScript frontend and the Python-based Agno framework
2. The API bridge is located at `src/app/api/agno/route.ts`
3. A TypeScript service (`src/lib/agno.ts`) provides helper functions for working with Agno

## Setup Requirements

To use Agno in this project, you'll need:

1. Python 3.8+ installed on your development and deployment environments
2. The project's virtual environment activated (`source agno_venv/bin/activate`)

## Virtual Environment

This project uses a Python virtual environment to isolate Agno dependencies:

```bash
# The virtual environment is already set up with:
python -m venv agno_venv

# Activate the environment before working with Agno:
source agno_venv/bin/activate

# The Agno library is installed in development mode from:
# pip install -e ./agno/libs/agno
```

Note: The virtual environment directory is excluded from git via `.gitignore`.

## Next Steps

- Implement Python scripts for agent-specific implementations
- Configure environment variables for Agno
- Set up proper error handling and logging for the Agno integration
- Create tests for the integration

## Reference

- [Agno Documentation](https://docs.agno.com)
- [CitifyAI Architecture Documentation](../cline_docs/codebaseSummary.md)
