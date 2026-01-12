@echo off
REM ================================================================
REM DISABLE CLOUDWATCH LOGS FOR FARGATE
REM ================================================================
REM This script updates your ECS task definition to remove CloudWatch logging
REM Saves you $$$ on CloudWatch PUT costs
REM ================================================================

echo.
echo ============================================================
echo   DISABLE CloudWatch Logs for ICDA Fargate
echo ============================================================
echo.

REM Set your values here
set CLUSTER_NAME=YOUR_CLUSTER_NAME
set SERVICE_NAME=YOUR_SERVICE_NAME
set TASK_FAMILY=icda-prototype
set REGION=us-east-1

echo [1/4] Getting current task definition...
aws ecs describe-task-definition --task-definition %TASK_FAMILY% --region %REGION% --query "taskDefinition" > current-task-def.json

echo [2/4] Creating new task definition WITHOUT CloudWatch logs...
REM The task-definition.json in this folder has NO logConfiguration = NO CloudWatch costs

echo [3/4] Registering new task definition...
aws ecs register-task-definition --cli-input-json file://task-definition.json --region %REGION%

echo [4/4] Updating service to use new task definition...
aws ecs update-service --cluster %CLUSTER_NAME% --service %SERVICE_NAME% --task-definition %TASK_FAMILY% --region %REGION%

echo.
echo ============================================================
echo   DONE! CloudWatch logging disabled.
echo   New tasks will NOT send logs to CloudWatch.
echo   Monitor via: docker logs, or add your own logging solution
echo ============================================================
echo.
pause
