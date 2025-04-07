import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from agno.memory_v2.memory import Memory, UserMemory, SessionSummary, TeamContext, TeamMemberInteraction
from agno.models.message import Message
from agno.run.response import RunResponse


@pytest.fixture
def mock_model():
    model = Mock()
    model.supports_native_structured_outputs = False
    model.supports_json_schema_outputs = False
    model.response_format = {"type": "json_object"}
    return model


@pytest.fixture
def memory_with_model(mock_model):
    return Memory(model=mock_model)


@pytest.fixture
def mock_memory_manager():
    manager = Mock()
    return manager


@pytest.fixture
def mock_summary_manager():
    manager = Mock()
    return manager


@pytest.fixture
def memory_with_managers(mock_model, mock_memory_manager, mock_summary_manager):
    return Memory(
        model=mock_model,
        memory_manager=mock_memory_manager,
        summarizer=mock_summary_manager
    )


@pytest.fixture
def sample_user_memory():
    return UserMemory(
        memory="The user's name is John Doe",
        topics=["name", "user"],
        last_updated=datetime.now()
    )


@pytest.fixture
def sample_session_summary():
    return SessionSummary(
        summary="This was a session about stocks",
        topics=["stocks", "finance"],
        last_updated=datetime.now()
    )


@pytest.fixture
def sample_run_response():
    return RunResponse(
        content="Sample response content",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ]
    )


# Memory Initialization Tests
def test_default_initialization():
    memory = Memory()
    assert memory.memories == {}
    assert memory.summaries == {}
    assert memory.runs == {}
    assert memory.team_context == {}
    assert memory.model is not None


def test_initialization_with_model(mock_model):
    memory = Memory(model=mock_model)
    assert memory.model == mock_model


def test_initialization_with_memories(sample_user_memory):
    memories = {"user1": {"memory1": sample_user_memory}}
    memory = Memory(memories=memories)
    assert memory.memories == memories
    assert memory.get_user_memory("user1", "memory1") == sample_user_memory


# User Memory Operations Tests
def test_add_user_memory(memory_with_model, sample_user_memory):
    memory_id = memory_with_model.add_user_memory(
        memory=sample_user_memory,
        user_id="test_user"
    )
    
    assert memory_id is not None
    assert memory_with_model.memories["test_user"][memory_id] == sample_user_memory
    assert memory_with_model.get_user_memory("test_user", memory_id) == sample_user_memory


def test_add_user_memory_default_user(memory_with_model, sample_user_memory):
    memory_id = memory_with_model.add_user_memory(memory=sample_user_memory)
    
    assert memory_id is not None
    assert memory_with_model.memories["default"][memory_id] == sample_user_memory
    assert memory_with_model.get_user_memory("default", memory_id) == sample_user_memory


def test_replace_user_memory(memory_with_model, sample_user_memory):
    # First add a memory
    memory_id = memory_with_model.add_user_memory(
        memory=sample_user_memory,
        user_id="test_user"
    )
    
    # Now replace it
    updated_memory = UserMemory(
        memory="The user's name is Jane Doe",
        topics=["name", "user"],
        last_updated=datetime.now()
    )
    
    memory_with_model.replace_user_memory(
        memory_id=memory_id,
        memory=updated_memory,
        user_id="test_user"
    )
    
    retrieved_memory = memory_with_model.get_user_memory("test_user", memory_id)
    assert retrieved_memory == updated_memory
    assert retrieved_memory.memory == "The user's name is Jane Doe"


def test_delete_user_memory(memory_with_model, sample_user_memory):
    # First add a memory
    memory_id = memory_with_model.add_user_memory(
        memory=sample_user_memory,
        user_id="test_user"
    )
    
    # Verify it exists
    assert memory_with_model.get_user_memory("test_user", memory_id) is not None
    
    # Now delete it
    memory_with_model.delete_user_memory("test_user", memory_id)
    
    # Verify it's gone
    assert memory_id not in memory_with_model.memories["test_user"]


def test_get_user_memories(memory_with_model, sample_user_memory):
    # Add two memories
    memory_with_model.add_user_memory(
        memory=sample_user_memory,
        user_id="test_user"
    )
    
    memory_with_model.add_user_memory(
        memory=UserMemory(memory="User likes pizza", topics=["food"]),
        user_id="test_user"
    )
    
    # Get all memories for the user
    memories = memory_with_model.get_user_memories("test_user")
    
    assert len(memories) == 2
    assert any(m.memory == "The user's name is John Doe" for m in memories)
    assert any(m.memory == "User likes pizza" for m in memories)


# Session Summary Operations Tests
def test_create_session_summary(memory_with_managers):
    # Setup the mock to return a summary
    mock_summary = MagicMock()
    mock_summary.summary = "Test summary"
    mock_summary.topics = ["test"]
    memory_with_managers.summary_manager.run.return_value = mock_summary
    
    # Add a run to have messages for the summary
    session_id = "test_session"
    user_id = "test_user"
    
    run_response = RunResponse(
        content="Sample response",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!")
        ]
    )
    
    memory_with_managers.add_run(session_id, run_response)
    
    # Create the summary
    summary = memory_with_managers.create_session_summary(session_id, user_id)
    
    assert summary is not None
    assert summary.summary == "Test summary"
    assert summary.topics == ["test"]
    assert memory_with_managers.summaries[user_id][session_id] == summary


def test_get_session_summary(memory_with_model, sample_session_summary):
    # Add a summary
    session_id = "test_session"
    user_id = "test_user"
    
    memory_with_model.summaries = {
        user_id: {
            session_id: sample_session_summary
        }
    }
    
    # Retrieve the summary
    summary = memory_with_model.get_session_summary(user_id, session_id)
    
    assert summary == sample_session_summary
    assert summary.summary == "This was a session about stocks"


def test_delete_session_summary(memory_with_model, sample_session_summary):
    # Add a summary
    session_id = "test_session"
    user_id = "test_user"
    
    memory_with_model.summaries = {
        user_id: {
            session_id: sample_session_summary
        }
    }
    
    # Verify it exists
    assert memory_with_model.get_session_summary(user_id, session_id) is not None
    
    # Now delete it
    memory_with_model.delete_session_summary(user_id, session_id)
    
    # Verify it's gone
    assert session_id not in memory_with_model.summaries[user_id]


# Memory Search Tests
def test_search_user_memories_semantic(memory_with_model):
    # Setup test data
    memory_with_model.memories = {
        "test_user": {
            "memory1": UserMemory(memory="Memory 1", topics=["topic1"]),
            "memory2": UserMemory(memory="Memory 2", topics=["topic2"])
        }
    }
    
    # Mock the internal method
    with patch.object(memory_with_model, '_search_user_memories_semantic') as mock_search:
        # Setup the mock to return a memory list
        mock_memories = [
            UserMemory(memory="Memory 1", topics=["topic1"]),
            UserMemory(memory="Memory 2", topics=["topic2"])
        ]
        mock_search.return_value = mock_memories
        
        # Call the search function
        results = memory_with_model.search_user_memories(
            query="test query",
            retrieval_method="semantic",
            user_id="test_user"
        )
        
        # Verify the search was called correctly
        mock_search.assert_called_once_with(user_id="test_user", query="test query", limit=None)
        assert results == mock_memories


def test_search_user_memories_last_n(memory_with_model):
    # Setup test data
    memory_with_model.memories = {
        "test_user": {
            "memory1": UserMemory(memory="Memory 1", topics=["topic1"]),
            "memory2": UserMemory(memory="Memory 2", topics=["topic2"])
        }
    }
    
    # Mock the internal method
    with patch.object(memory_with_model, '_get_last_n_memories') as mock_search:
        # Setup the mock to return a memory list
        mock_memories = [
            UserMemory(memory="Recent Memory 1", topics=["topic1"]),
            UserMemory(memory="Recent Memory 2", topics=["topic2"])
        ]
        mock_search.return_value = mock_memories
        
        # Call the search function
        results = memory_with_model.search_user_memories(
            retrieval_method="last_n",
            limit=2,
            user_id="test_user"
        )
        
        # Verify the search was called correctly
        mock_search.assert_called_once_with(user_id="test_user", limit=2)
        assert results == mock_memories


def test_search_user_memories_first_n(memory_with_model):
    # Setup test data
    memory_with_model.memories = {
        "test_user": {
            "memory1": UserMemory(memory="Memory 1", topics=["topic1"]),
            "memory2": UserMemory(memory="Memory 2", topics=["topic2"])
        }
    }
    
    # Mock the internal method
    with patch.object(memory_with_model, '_get_first_n_memories') as mock_search:
        # Setup the mock to return a memory list
        mock_memories = [
            UserMemory(memory="Old Memory 1", topics=["topic1"]),
            UserMemory(memory="Old Memory 2", topics=["topic2"])
        ]
        mock_search.return_value = mock_memories
        
        # Call the search function
        results = memory_with_model.search_user_memories(
            retrieval_method="first_n",
            limit=2,
            user_id="test_user"
        )
        
        # Verify the search was called correctly
        mock_search.assert_called_once_with(user_id="test_user", limit=2)
        assert results == mock_memories


# Run and Messages Tests
def test_add_run(memory_with_model, sample_run_response):
    session_id = "test_session"
    
    # Add a run
    memory_with_model.add_run(session_id, sample_run_response)
    
    # Verify it was added
    assert session_id in memory_with_model.runs
    assert len(memory_with_model.runs[session_id]) == 1
    assert memory_with_model.runs[session_id][0] == sample_run_response


def test_get_messages_for_session(memory_with_model, sample_run_response):
    session_id = "test_session"
    
    # Add a run
    memory_with_model.add_run(session_id, sample_run_response)
    
    # Get messages
    messages = memory_with_model.get_messages_for_session(session_id)
    
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Hello"
    assert messages[1].role == "assistant"
    assert messages[1].content == "Hi there!"


def test_get_messages_from_last_n_runs(memory_with_model):
    session_id = "test_session"
    
    # Add multiple runs
    run1 = RunResponse(
        content="First response",
        messages=[
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer")
        ]
    )
    
    run2 = RunResponse(
        content="Second response",
        messages=[
            Message(role="user", content="Second question"),
            Message(role="assistant", content="Second answer")
        ]
    )
    
    memory_with_model.add_run(session_id, run1)
    memory_with_model.add_run(session_id, run2)
    
    # Get last run's messages
    messages = memory_with_model.get_messages_from_last_n_runs(session_id, last_n=1)
    
    assert len(messages) == 2
    assert messages[0].content == "Second question"
    assert messages[1].content == "Second answer"


# Team Context Tests
def test_add_interaction_to_team_context(memory_with_model, sample_run_response):
    session_id = "team_session"
    member_name = "Agent1"
    task = "Research task"
    
    # Add an interaction
    memory_with_model.add_interaction_to_team_context(session_id, member_name, task, sample_run_response)
    
    # Verify it was added
    assert session_id in memory_with_model.team_context
    assert len(memory_with_model.team_context[session_id].member_interactions) == 1
    
    interaction = memory_with_model.team_context[session_id].member_interactions[0]
    assert interaction.member_name == member_name
    assert interaction.task == task
    assert interaction.response == sample_run_response


def test_set_team_context_text(memory_with_model):
    session_id = "team_session"
    context_text = "This is team context information"
    
    # Set context text
    memory_with_model.set_team_context_text(session_id, context_text)
    
    # Verify it was set
    assert session_id in memory_with_model.team_context
    assert memory_with_model.team_context[session_id].text == context_text
    
    # Test get_team_context_str
    context_str = memory_with_model.get_team_context_str(session_id)
    assert context_text in context_str


def test_team_context_with_multiple_interactions(memory_with_model):
    session_id = "team_session"
    
    # Add multiple interactions
    run1 = RunResponse(
        content="Research result",
        messages=[Message(role="assistant", content="Research findings")]
    )
    
    run2 = RunResponse(
        content="Analysis result",
        messages=[Message(role="assistant", content="Analysis complete")]
    )
    
    memory_with_model.add_interaction_to_team_context(session_id, "Researcher", "Do research", run1)
    memory_with_model.add_interaction_to_team_context(session_id, "Analyst", "Analyze data", run2)
    
    # Verify interactions were added
    assert session_id in memory_with_model.team_context
    assert len(memory_with_model.team_context[session_id].member_interactions) == 2


# Memory Integration Tests
def test_create_user_memories(memory_with_managers):
    # Setup mock response
    mock_updates = [
        MagicMock(id=None, memory="New memory 1", topics=["topic1"]),
        MagicMock(id=None, memory="New memory 2", topics=["topic2"])
    ]
    memory_with_managers.memory_manager.run.return_value = MagicMock(updates=mock_updates)
    
    # Create user memories
    messages = [Message(role="user", content="Remember this information")]
    result = memory_with_managers.create_user_memories(messages, user_id="test_user")
    
    # Verify memories were created
    assert len(result) == 2
    assert "test_user" in memory_with_managers.memories
    assert len(memory_with_managers.memories["test_user"]) == 2
    
    memories = memory_with_managers.get_user_memories("test_user")
    assert any(m.memory == "New memory 1" for m in memories)
    assert any(m.memory == "New memory 2" for m in memories)


def test_to_dict_and_from_dict(memory_with_model, sample_user_memory, sample_session_summary):
    # Setup memory with user memories and summaries
    user_id = "test_user"
    memory_id = memory_with_model.add_user_memory(sample_user_memory, user_id=user_id)
    
    session_id = "test_session"
    memory_with_model.summaries = {
        user_id: {
            session_id: sample_session_summary
        }
    }
    
    # Get dictionary representation
    memory_dict = memory_with_model.to_dict()
    
    # Verify the dictionary contains our data
    assert "memories" in memory_dict
    assert "summaries" in memory_dict
    assert user_id in memory_dict["memories"]
    assert user_id in memory_dict["summaries"]
    assert session_id in memory_dict["summaries"][user_id]
    
    # Create a new memory from the dictionary
    new_memory = Memory()
    new_memory.memories = {user_id: {memory_id: UserMemory.from_dict(memory) for memory_id, memory in memory_dict.get("memories", {}).get(user_id, {}).items()}}
    new_memory.summaries = {user_id: {session_id: SessionSummary.from_dict(summary) for session_id, summary in memory_dict.get("summaries", {}).get(user_id, {}).items()}}
    
    # Verify the new memory has the same data
    assert memory_id in new_memory.memories[user_id]
    assert new_memory.memories[user_id][memory_id].memory == sample_user_memory.memory
    assert new_memory.summaries[user_id][session_id].summary == sample_session_summary.summary


def test_clear(memory_with_model, sample_user_memory):
    # Add data to memory
    memory_with_model.add_user_memory(sample_user_memory, user_id="test_user")
    memory_with_model.summaries = {
        "test_user": {
            "test_session": SessionSummary(summary="Test summary")
        }
    }
    
    # Clear memory
    memory_with_model.clear()
    
    # Verify data is cleared
    assert memory_with_model.memories == {}
    assert memory_with_model.summaries == {}


if __name__ == "__main__":
    pytest.main()
