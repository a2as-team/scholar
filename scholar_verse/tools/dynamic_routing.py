"""Enhanced dynamic routing module for ScholarVerse with ADK integration.

This module provides dynamic routing capabilities for the ScholarVerse router agent,
leveraging Google ADK tools and the enhanced state management system to intelligently
route requests to the appropriate sub-agents based on content analysis, workflow stage,
and user preferences.
"""

from typing import Dict, Any, List, Optional, Type, TypeVar, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from functools import wraps

from google.adk.tools import ToolContext, tool
from google.adk.tools.agent_tool import AgentTool
from google.adk.core.exceptions import ToolExecutionError

from scholar_verse.shared_libraries.logging_utils import logger
from scholar_verse.shared_libraries.state_management.adaptive_state import (
    AdaptiveStateManager, WorkflowStage, WorkflowState, DocumentState
)

# Type variable for agent tools
AgentToolT = TypeVar('AgentToolT', bound=AgentTool)

# Import sub-agents
from scholar_verse.sub_agents.ingestion.agent import ingestion_agent_tool
from scholar_verse.sub_agents.citation_graph.agent import citation_graph_agent_tool
from scholar_verse.sub_agents.cross_paper_analysis.agent import cross_paper_analysis_agent_tool
from scholar_verse.sub_agents.deep_search.agent import deep_search_agent_tool
from scholar_verse.sub_agents.insight.agent import insight_agent_tool
from scholar_verse.sub_agents.visualization.agent import visualization_agent_tool

# Type aliases
AgentResponse = Dict[str, Any]
AgentCallable = Callable[..., Awaitable[AgentResponse]]

# Agent registry mapping agent names to their tool instances
AGENT_REGISTRY: Dict[str, AgentTool] = {
    'ingestion': ingestion_agent_tool,
    'citation_graph': citation_graph_agent_tool,
    'cross_paper_analysis': cross_paper_analysis_agent_tool,
    'deep_search': deep_search_agent_tool,
    'insight': insight_agent_tool,
    'visualization': visualization_agent_tool
}

# Workflow stage transitions
WORKFLOW_TRANSITIONS: Dict[WorkflowStage, List[WorkflowStage]] = {
    WorkflowStage.DOCUMENT_INGESTION: [
        WorkflowStage.CITATION_GRAPH,
        WorkflowStage.DEEP_SEARCH
    ],
    WorkflowStage.CITATION_GRAPH: [
        WorkflowStage.CROSS_PAPER_ANALYSIS
    ],
    WorkflowStage.DEEP_SEARCH: [
        WorkflowStage.CROSS_PAPER_ANALYSIS
    ],
    WorkflowStage.CROSS_PAPER_ANALYSIS: [
        WorkflowStage.INSIGHT_GENERATION
    ],
    WorkflowStage.INSIGHT_GENERATION: [
        WorkflowStage.VISUALIZATION
    ],
    WorkflowStage.VISUALIZATION: []
}


@dataclass
class WorkflowStageConfig:
    """Configuration for a workflow stage."""
    description: str
    primary_agent: AgentTool
    required_capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'description': self.description,
            'agent_name': self.primary_agent.name,
            'required_capabilities': self.required_capabilities
        }

def get_agent_for_stage(stage: WorkflowStage) -> Optional[AgentTool]:
    """Get the primary agent tool for a workflow stage.
    
    Args:
        stage: The workflow stage
        
    Returns:
        The agent tool for the stage, or None if not found
    """
    if stage in WORKFLOW_STAGES:
        return WORKFLOW_STAGES[stage].primary_agent
    return None

def get_available_agents_for_stage(stage: WorkflowStage) -> List[AgentTool]:
    """Get all available agents that can handle a workflow stage.
    
    Args:
        stage: The workflow stage
        
    Returns:
        List of available agent tools
    """
    agents = []
    if stage in WORKFLOW_STAGES:
        primary_agent = WORKFLOW_STAGES[stage].primary_agent
        if primary_agent:
            agents.append(primary_agent)
    return agents

# Define workflow stages with enhanced metadata
WORKFLOW_STAGES: Dict[WorkflowStage, WorkflowStageConfig] = {
    WorkflowStage.DOCUMENT_INGESTION: WorkflowStageConfig(
        description='Process and extract text from PDF documents',
        primary_agent=ingestion_agent_tool,
        required_capabilities=['document_parsing', 'text_extraction']
    ),
    WorkflowStage.CITATION_GRAPH: WorkflowStageConfig(
        description='Construct the citation graph in Neo4j',
        primary_agent=citation_graph_agent_tool,
        required_capabilities=['graph_construction', 'citation_parsing']
    ),
    WorkflowStage.DEEP_SEARCH: WorkflowStageConfig(
        description='Conduct web search for additional information',
        primary_agent=deep_search_agent_tool,
        required_capabilities=['web_search', 'content_retrieval']
    ),
    WorkflowStage.CROSS_PAPER_ANALYSIS: WorkflowStageConfig(
        description='Analyze relationships between multiple papers',
        primary_agent=cross_paper_analysis_agent_tool,
        required_capabilities=['semantic_analysis', 'relationship_extraction']
    ),
    WorkflowStage.INSIGHT_GENERATION: WorkflowStageConfig(
        description='Generate insights from the analysis',
        primary_agent=insight_agent_tool,
        required_capabilities=['summarization', 'insight_generation']
    ),
    WorkflowStage.VISUALIZATION: WorkflowStageConfig(
        description='Create visualizations of the knowledge graph',
        primary_agent=visualization_agent_tool,
        required_capabilities=['data_visualization', 'graph_rendering']
    ),
}

@tool
def get_workflow_status(context: ToolContext) -> Dict[str, Any]:
    """Get the current status of the workflow.
    
    Args:
        context: The tool context
        
    Returns:
        Dictionary containing workflow status information
    """
    state_manager = AdaptiveStateManager()
    current_state = state_manager.get(context)
    workflow_state = state_manager.get_workflow_state(context)
    
    return {
        'current_stage': workflow_state.current_stage.value if workflow_state.current_stage else None,
        'stages_completed': [s.value for s in workflow_state.stages_completed],
        'next_possible_stages': [s.value for s in workflow_state.next_possible_stages],
        'last_updated': current_state.get('metadata', {}).get('last_updated')
    }

@tool
async def route_to_agent(
    context: ToolContext,
    agent_name: str,
    input_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> AgentResponse:
    """Route a request to a specific agent with input validation.
    
    Args:
        context: The tool context
        agent_name: Name of the agent to route to
        input_data: Input data for the agent
        metadata: Optional metadata for the request
        
    Returns:
        Response from the agent
    """
    logger.info(f"Routing to agent: {agent_name}")
    
    # Get the agent tool
    agent_tool = AGENT_REGISTRY.get(agent_name)
    if not agent_tool:
        raise ToolExecutionError(f"Unknown agent: {agent_name}")
    
    # Get current workflow state
    state_manager = AdaptiveStateManager()
    workflow_state = state_manager.get_workflow_state(context)
    
    # Validate agent is appropriate for current stage
    current_stage = workflow_state.current_stage
    if current_stage and agent_tool not in get_available_agents_for_stage(current_stage):
        raise ToolExecutionError(
            f"Agent {agent_name} is not available for stage {current_stage.value}"
        )
    
    try:
        # Execute the agent with input data
        response = await agent_tool.run_async(
            args=input_data,
            tool_context=context
        )
        
        # Update state with agent response
        state_manager.update_document_state(context, {
            'last_agent': agent_name,
            'last_agent_response': response
        })
        
        # Update workflow stage if needed
        if current_stage and current_stage not in workflow_state.stages_completed:
            workflow_state = state_manager.update_workflow_stage(
                context, current_stage, is_complete=True
            )
        
        return {
            'success': True,
            'agent': agent_name,
            'response': response,
            'next_steps': workflow_state.next_possible_stages
        }
        
    except Exception as e:
        logger.error(f"Error in agent {agent_name}: {str(e)}")
        raise ToolExecutionError(f"Error executing agent {agent_name}: {str(e)}")

@tool
@tool
async def analyze_request(
    context: ToolContext,
    user_input: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Analyze a user request to determine the appropriate workflow stage and agent.
    
    This function uses the current workflow state and the user's input to determine
    the most appropriate next steps in the workflow.
    
    Args:
        context: The tool context
        user_input: The user's input text
        metadata: Optional metadata for the analysis
        
    Returns:
        Dictionary with analysis results and recommended actions
    """
    logger.info(f"Analyzing request: {user_input[:100]}...")
    state_manager = AdaptiveStateManager()
    workflow_state = state_manager.get_workflow_state(context)
    
    # Get current stage and available agents
    current_stage = workflow_state.current_stage
    available_agents = get_available_agents_for_stage(current_stage) if current_stage else []
    
    # Determine next possible stages based on current state
    next_stages = workflow_state.next_possible_stages
    
    # In a real implementation, you would use an LLM or other NLP to analyze the request
    # and determine the most appropriate next step. For now, we'll use a simple approach.
    
    # If we have a current stage, recommend its primary agent
    recommended_agent = None
    if current_stage and current_stage in WORKFLOW_STAGES:
        recommended_agent = WORKFLOW_STAGES[current_stage].primary_agent.name
    
    return {
        'analysis': {
            'intent': 'process_document' if not current_stage else 'continue_workflow',
            'confidence': 0.9,
            'entities': {}
        },
        'recommendations': {
            'next_stages': [stage.value for stage in next_stages],
            'recommended_agent': recommended_agent,
            'available_agents': [agent.name for agent in available_agents]
        },
        'current_state': {
            'current_stage': current_stage.value if current_stage else None,
            'stages_completed': [s.value for s in workflow_state.stages_completed]
        }
    }

@tool
async def execute_workflow_stage(
    context: ToolContext,
    stage: WorkflowStage,
    input_data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute a complete workflow stage with the provided input data.
    
    This is a higher-level function that handles the execution of an entire
    workflow stage, including input validation, agent execution, and state updates.
    
    Args:
        context: The tool context
        stage: The workflow stage to execute
        input_data: Input data required for the stage
        metadata: Optional metadata for the execution
        
    Returns:
        Dictionary with execution results
    """
    logger.info(f"Executing workflow stage: {stage.value}")
    
    # Get the primary agent for this stage
    agent_tool = get_agent_for_stage(stage)
    if not agent_tool:
        raise ToolExecutionError(f"No agent available for stage: {stage.value}")
    
    # Update workflow state to the new stage
    state_manager = AdaptiveStateManager()
    state_manager.update_workflow_stage(context, stage, is_complete=False)
    
    # Execute the agent
    try:
        response = await route_to_agent(
            context=context,
            agent_name=agent_tool.name,
            input_data=input_data,
            metadata=metadata
        )
        
        # Mark stage as complete if execution was successful
        state_manager.update_workflow_stage(context, stage, is_complete=True)
        
        # Get updated workflow state
        workflow_state = state_manager.get_workflow_state(context)
        
        return {
            'success': True,
            'stage': stage.value,
            'response': response,
            'next_possible_stages': [s.value for s in workflow_state.next_possible_stages]
        }
        
    except Exception as e:
        logger.error(f"Error executing workflow stage {stage.value}: {str(e)}")
        raise ToolExecutionError(f"Failed to execute workflow stage {stage.value}: {str(e)}")

@tool
def get_available_workflow_stages() -> Dict[str, Any]:
    """Get information about all available workflow stages.
    
    Returns:
        Dictionary containing information about all workflow stages
    """
    return {
        stage.value: {
            'description': config.description,
            'required_capabilities': config.required_capabilities,
            'primary_agent': config.primary_agent.name if config.primary_agent else None
        }
        for stage, config in WORKFLOW_STAGES.items()
    }

@tool
def get_available_agents() -> Dict[str, Any]:
    """Get information about all available agents.
    
    Returns:
        Dictionary containing information about all registered agents
    """
    return {
        name: {
            'description': tool.description,
            'input_schema': tool.input_schema,
            'output_schema': tool.output_schema
        }
        for name, tool in AGENT_REGISTRY.items()
    }


async def route_to_agent(
    request: str,
    agent_name: str,
    tool_context: ToolContext,
) -> str:
    """Route a request to a specific sub-agent.
    
    Args:
        request: The user request to route.
        agent_name: The name of the agent to route to.
        tool_context: The tool context containing state and other information.
        
    Returns:
        The response from the sub-agent.
    """
    logger.info(f"Routing request to {agent_name} agent")
    
    # Get the state manager
    state_manager = tool_context.state.get('state_manager')
    if not state_manager:
        state_manager = AdaptiveStateManager()
        tool_context.state['state_manager'] = state_manager
    
    # Get the current state
    current_state = state_manager.get(tool_context)
    
    # Map agent name to agent tool
    agent_tools = {
        'ingestion': ingestion_agent_tool,
        'citation_graph': citation_graph_agent_tool,
        'cross_paper_analysis': cross_paper_analysis_agent_tool,
        'deep_search': deep_search_agent_tool,
        'insight': insight_agent_tool,
        'visualization': visualization_agent_tool,
    }
    
    agent_tool = agent_tools.get(agent_name)
    if not agent_tool:
        logger.error(f"Unknown agent: {agent_name}")
        return f"Error: Unknown agent '{agent_name}'"
    
    # Update the workflow stage based on the agent
    for stage, info in WORKFLOW_STAGES.items():
        if info['primary_agent'] == agent_tool:
            state_manager.update_workflow_stage(stage)
            break
    
    # Call the agent
    try:
        response = await agent_tool.run_async(
            args={"request": request}, 
            tool_context=tool_context
        )
        
        # Store the agent's response in the state
        current_state = state_manager.get(tool_context)
        if 'agent_responses' not in current_state:
            current_state['agent_responses'] = {}
        current_state['agent_responses'][agent_name] = response
        state_manager.set(tool_context, current_state)
        
        # Update agent performance metrics
        state_manager.update_agent_performance(agent_name, {
            'last_used': tool_context.state.get('timestamp', ''),
            'success': True,
        })
        
        return response
    except Exception as e:
        logger.error(f"Error calling {agent_name} agent: {str(e)}")
        
        # Update agent performance metrics with error
        state_manager.update_agent_performance(agent_name, {
            'last_used': tool_context.state.get('timestamp', ''),
            'success': False,
            'error': str(e),
        })
        
        return f"Error calling {agent_name} agent: {str(e)}"


async def analyze_request(
    request: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """Analyze a user request to determine the appropriate workflow and agents.
    
    Args:
        request: The user request to analyze.
        tool_context: The tool context containing state and other information.
        
    Returns:
        A dictionary containing the analysis results.
    """
    logger.info(f"Analyzing request: {request}")
    
    # Get the state manager
    state_manager = tool_context.state.get('state_manager')
    if not state_manager:
        state_manager = AdaptiveStateManager()
        tool_context.state['state_manager'] = state_manager
    
    # Get the current state
    current_state = state_manager.get(tool_context)
    
    # Simple keyword-based analysis for now
    # This will be enhanced with more sophisticated NLP in future phases
    analysis = {
        'document_related': any(kw in request.lower() for kw in ['pdf', 'document', 'paper', 'upload', 'read']),
        'citation_related': any(kw in request.lower() for kw in ['citation', 'cite', 'reference', 'graph', 'network']),
        'search_related': any(kw in request.lower() for kw in ['search', 'find', 'look up', 'google', 'web']),
        'analysis_related': any(kw in request.lower() for kw in ['analyze', 'analysis', 'compare', 'comparison', 'relationship']),
        'insight_related': any(kw in request.lower() for kw in ['insight', 'summarize', 'summary', 'recommend', 'recommendation']),
        'visualization_related': any(kw in request.lower() for kw in ['visualize', 'visualization', 'display', 'show', 'graph', '3d']),
    }
    
    # Determine the most appropriate agent based on the analysis
    recommended_agents = []
    if analysis['document_related']:
        recommended_agents.append('ingestion')
    if analysis['citation_related']:
        recommended_agents.append('citation_graph')
    if analysis['search_related']:
        recommended_agents.append('deep_search')
    if analysis['analysis_related']:
        recommended_agents.append('cross_paper_analysis')
    if analysis['insight_related']:
        recommended_agents.append('insight')
    if analysis['visualization_related']:
        recommended_agents.append('visualization')
    
    # If no specific agents are recommended, use the current workflow stage
    if not recommended_agents and current_state['workflow']['current_stage']:
        current_stage = current_state['workflow']['current_stage']
        for agent_name, agent_tool in {
            'ingestion': ingestion_agent_tool,
            'citation_graph': citation_graph_agent_tool,
            'cross_paper_analysis': cross_paper_analysis_agent_tool,
            'deep_search': deep_search_agent_tool,
            'insight': insight_agent_tool,
            'visualization': visualization_agent_tool,
        }.items():
            if WORKFLOW_STAGES.get(current_stage, {}).get('primary_agent') == agent_tool:
                recommended_agents.append(agent_name)
                break
    
    # Store the analysis in the state
    if 'request_analysis' not in current_state:
        current_state['request_analysis'] = {}
    current_state['request_analysis'][request] = {
        'analysis': analysis,
        'recommended_agents': recommended_agents,
    }
    state_manager.set(tool_context, current_state)
    
    return {
        'analysis': analysis,
        'recommended_agents': recommended_agents,
    }


async def execute_workflow(
    request: str,
    workflow_stage: str,
    tool_context: ToolContext,
) -> str:
    """Execute a workflow stage for a given request.
    
    Args:
        request: The user request to process.
        workflow_stage: The workflow stage to execute.
        tool_context: The tool context containing state and other information.
        
    Returns:
        The response from the workflow execution.
    """
    logger.info(f"Executing workflow stage: {workflow_stage}")
    
    # Get the state manager
    state_manager = tool_context.state.get('state_manager')
    if not state_manager:
        state_manager = AdaptiveStateManager()
        tool_context.state['state_manager'] = state_manager
    
    # Validate the workflow stage
    if workflow_stage not in WORKFLOW_STAGES:
        logger.error(f"Unknown workflow stage: {workflow_stage}")
        return f"Error: Unknown workflow stage '{workflow_stage}'"
    
    # Update the current workflow stage
    state_manager.update_workflow_stage(workflow_stage)
    
    # Get the primary agent for this stage
    primary_agent = WORKFLOW_STAGES[workflow_stage]['primary_agent']
    
    # Map agent tool to agent name
    agent_name_map = {
        ingestion_agent_tool: 'ingestion',
        citation_graph_agent_tool: 'citation_graph',
        cross_paper_analysis_agent_tool: 'cross_paper_analysis',
        deep_search_agent_tool: 'deep_search',
        insight_agent_tool: 'insight',
        visualization_agent_tool: 'visualization',
    }
    
    agent_name = agent_name_map.get(primary_agent, 'unknown')
    
    # Route the request to the primary agent
    response = await route_to_agent(request, agent_name, tool_context)
    
    return response
