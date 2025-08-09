import express from 'express';
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNodeHttpEndpoint,
  langGraphPlatformEndpoint,
  LangGraphAgent,
} from '@copilotkit/runtime';
import cors from "cors";

const corsOrigins =["http://localhost:3001"]

const app = express();
app.use(
  cors({
    origin: corsOrigins,
    credentials: true,
  }),
);
const serviceAdapter = new ExperimentalEmptyAdapter();

const runtime = new CopilotRuntime({
   remoteEndpoints: [
    langGraphPlatformEndpoint({
      deploymentUrl:  'http://127.0.0.1:2024',
      graphId: 'app',
      agents: [{
          name: 'app',
          description: 'A helpful LLM agent.'
      }]
    }),
  ],
});

const handler = copilotRuntimeNodeHttpEndpoint({
  endpoint: '/copilotkit',
  runtime,
  serviceAdapter,
});

app.use('/copilotkit', handler);

app.use('/health', (req, res) => {
  res.send('OK');
})

app.listen(3001, () => {
  console.log('Listening at http://localhost:3001/copilotkit');
});