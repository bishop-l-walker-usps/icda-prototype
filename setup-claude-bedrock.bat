@echo off
REM ============================================================================
REM Claude Code - AWS Bedrock Configuration
REM ============================================================================
REM This script switches Claude Code from Anthropic API to AWS Bedrock
REM Run this before starting Claude Code to use your AWS account
REM ============================================================================

REM Enable Bedrock integration (REQUIRED)
set CLAUDE_CODE_USE_BEDROCK=1

REM AWS Region (REQUIRED - Claude Code doesn't read from .aws config)
set AWS_REGION=us-east-1

REM ============================================================================
REM AUTHENTICATION - Choose ONE method by uncommenting the appropriate section
REM ============================================================================

REM --- Option 1: AWS Profile (Recommended if you have AWS CLI configured) ---
set AWS_PROFILE=NNGC

REM --- Option 2: Access Keys (Uncomment and fill in if using direct keys) ---
REM set AWS_ACCESS_KEY_ID=your-access-key-id
REM set AWS_SECRET_ACCESS_KEY=your-secret-access-key
REM set AWS_SESSION_TOKEN=your-session-token

REM --- Option 3: Bedrock API Key (Uncomment if using Bedrock API key) ---
REM set AWS_BEARER_TOKEN_BEDROCK=your-bedrock-api-key

REM ============================================================================
REM MODEL CONFIGURATION (Optional - uses sensible defaults)
REM ============================================================================

REM Primary model (Claude Sonnet 4.5 - default)
REM set ANTHROPIC_MODEL=us.anthropic.claude-sonnet-4-5-20250929-v1:0

REM Fast/small model for quick tasks (Claude Haiku 4.5 - default)
REM set ANTHROPIC_SMALL_FAST_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0

REM Override region for small model if needed
REM set ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION=us-west-2

REM ============================================================================
REM PERFORMANCE SETTINGS (Recommended)
REM ============================================================================

REM Minimum recommended for Bedrock throttling logic
set CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384

REM Thinking tokens for extended reasoning
set MAX_THINKING_TOKENS=10240

REM Disable prompt caching if not available in your region
REM set DISABLE_PROMPT_CACHING=1

REM ============================================================================
REM VERIFICATION
REM ============================================================================

echo.
echo ============================================
echo  Claude Code - AWS Bedrock Configuration
echo ============================================
echo.
echo Configuration Applied:
echo   CLAUDE_CODE_USE_BEDROCK = %CLAUDE_CODE_USE_BEDROCK%
echo   AWS_REGION              = %AWS_REGION%
echo   AWS_PROFILE             = %AWS_PROFILE%
echo   MAX_OUTPUT_TOKENS       = %CLAUDE_CODE_MAX_OUTPUT_TOKENS%
echo   MAX_THINKING_TOKENS     = %MAX_THINKING_TOKENS%
echo.
echo ============================================
echo.
echo To verify Bedrock is configured, run:
echo   claude --info
echo.
echo To start Claude Code:
echo   claude
echo.
echo ============================================
echo.

REM Keep the shell open so environment variables persist
cmd /k
