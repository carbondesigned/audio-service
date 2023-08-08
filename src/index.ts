import express, {Express, Request, Response} from 'express';
import dotenv from 'dotenv';
import ytdl from 'ytdl-core';
import fs from 'fs';
import cp from 'child_process';
import {createClient} from '@supabase/supabase-js';
import {HfInference} from '@huggingface/inference';
import fetch from 'node-fetch';
import {OpenAI} from 'langchain/llms/openai';
import {PromptTemplate} from 'langchain/prompts';
import {LLMChain} from 'langchain/chains';
import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter';

// @ts-ignore
global.fetch = fetch;

dotenv.config();

const app: Express = express();
app.use(express.json());
const port = process.env.PORT;

export const supabaseClient = async (supabaseAccessToken: string) => {
  const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL as string;
  const SUPABASE_KEY = process.env.NEXT_PUBLIC_SUPABASE_KEY as string;
  const supabase = createClient(SUPABASE_URL, SUPABASE_KEY, {
    auth: {persistSession: false},
    global: {headers: {Authorization: `Bearer ${supabaseAccessToken}`}},
  });
  // set Supabase JWT on the client object,
  // so it is sent up with all Supabase requests
  return supabase;
};

app.get('/', (req: Request, res: Response) => {
  res.send('Express + TypeScript Server');
});

app.post('/api/video', async (req, res) => {
  const {video, token, userId, userEmail} = req.body;
  const supabase = await supabaseClient(token);
  let videoId = '';

  async function uploadFile(filePath: string) {
    const supabase = await supabaseClient(token);
    const file = fs.readFileSync(filePath);
    let {data, error} = await supabase.storage
      .from('chunks')
      .upload(filePath, file);
    if (error) throw error;
    return data;
  }

  ytdl
    .getInfo(video)
    .then((info) => {
      const format = ytdl.chooseFormat(info.formats, {quality: 'highestaudio'});
      const start = Date.now();

      videoId = info.videoDetails.videoId;
      const outputDir = `chunks/${videoId}`;
      fs.mkdirSync(outputDir, {recursive: true}); // create the directory if it doesn't exist

      const ffmpegProcess = cp.spawn(
        'ffmpeg',
        [
          '-i',
          'pipe:3',
          '-map',
          '0:a', // map only audio stream
          '-f',
          'segment',
          '-segment_time',
          '15', // split every 30 seconds
          '-c:a',
          'libmp3lame', // use libmp3lame codec for audio
          `${outputDir}/output%03d.mp3`, // output pattern
        ],
        {
          windowsHide: true,
          stdio: [
            /* Standard: stdin, stdout, stderr */
            'inherit',
            'inherit',
            'inherit',
            /* Custom: pipe:3, pipe:4 */
            'pipe',
            'pipe',
          ],
        }
      );

      ffmpegProcess.on('close', async () => {
        console.log(`\ndone, thanks - ${(Date.now() - start) / 1000}s`);

        // Upload files to Supabase after ffmpeg process is done
        const uploadPromises = [];
        for (let i = 0; i < 1000; i++) {
          // assuming a maximum of 1000 files
          const filePath = `${outputDir}/output${i
            .toString()
            .padStart(3, '0')}.mp3`;
          console.log(filePath);
          if (fs.existsSync(filePath)) {
            uploadPromises.push(uploadFile(filePath));
          } else {
            break; // stop the loop if file doesn't exist
          }
        }

        try {
          await Promise.all(uploadPromises);
        } catch (error) {
          console.log(error);
        }

        // After all files are uploaded, delete them
        for (let i = 0; i < uploadPromises.length; i++) {
          const filePath = `${outputDir}/output${i
            .toString()
            .padStart(3, '0')}.mp3`;
          if (fs.existsSync(filePath)) {
            fs.unlinkSync(filePath);
          }
        }

        // Check if the user already exists
        const {data: existingUser, error: userError} = await supabase
          .from('users')
          .select('userId')
          .eq('userId', userId);

        if (userError) throw userError;

        // If the user doesn't exist, insert a new user
        if (!existingUser.length) {
          const {data: newUser, error: insertError} = await supabase
            .from('users')
            .insert([
              {
                userId: userId,
                email: userEmail,
                // add other user fields here
              },
            ]);

          if (insertError) throw insertError;
        }

        // Associate the video with the user
        const {data: existingVideo, error: videoError} = await supabase
          .from('videos')
          .select('videoId')
          .eq('videoId', videoId);

        if (videoError) throw videoError;

        // If the video doesn't exist, insert a new video
        if (!existingVideo.length) {
          const {data: newVideo, error: insertError} = await supabase
            .from('videos')
            .insert([
              {
                videoId: videoId,
                // add other video fields here
              },
            ]);

          if (insertError) throw insertError;
        }

        const {data: existingAssociation, error: associationError} =
          await supabase
            .from('user_videos')
            .select('*')
            .eq('videoId', videoId)
            .eq('userId', userId);

        if (associationError) throw associationError;

        // If the association doesn't exist, create a new one
        if (!existingAssociation.length) {
          const {data, error: insertError} = await supabase
            .from('user_videos')
            .insert([
              {
                videoId: videoId,
                userId: userId, // replace with the user's id
              },
            ]);

          if (insertError) throw insertError;
        }
      });

      ytdl
        .downloadFromInfo(info, {format: format})
        .pipe(ffmpegProcess.stdio[3] as any);
    })
    .catch((error) => {
      console.log(error);
    })
    .finally(() => {
      // send message when done
      res.json({message: 'done'});
    });
});
app.post('/api/asr', async (req, res) => {
  const Hf = new HfInference(process.env.HUGGINGFACE_API_KEY);
  const {videoId, token} = req.body;
  const supabase = await supabaseClient(token);

  // get list of files in the videoId folder
  const {data: list, error: listError} = await supabase.storage
    .from('chunks')
    .list(`chunks/${videoId}`);

  if (listError) throw listError;

  let fullTranscription = '';

  for (const file of list) {
    // download the file
    const {data: downloadData, error: downloadError} = await supabase.storage
      .from('chunks')
      .download(`chunks/${videoId}/${file.name}`);

    if (downloadError) throw downloadError;

    // convert Blob to ArrayBuffer
    const arrayBuffer = await downloadData.arrayBuffer();

    try {
      const text = await Hf.automaticSpeechRecognition({
        model: 'jonatasgrosman/wav2vec2-large-xlsr-53-english',
        data: arrayBuffer,
      });

      fullTranscription += text.text + '\n';
      console.log('Transcription: ', text.text);
    } catch (error) {
      console.log(error);
    }
  }

  // upload full transcription to Supabase
  const {error: uploadError} = await supabase.storage
    .from('transcriptions')
    .upload(`${videoId}/transcription.txt`, Buffer.from(fullTranscription));

  if (uploadError) throw uploadError;

  // Associate the video with the user
  const {data: existingVideo, error: videoError} = await supabase
    .from('videos')
    .select('videoId')
    .eq('videoId', videoId);

  if (videoError) throw videoError;

  // If the video now has a transcription, delete the video's chunks
  if (fullTranscription) {
    for (const file of list) {
      const {error: deleteError} = await supabase.storage
        .from('chunks')
        .remove([`chunks/${videoId}/${file.name}`]);

      if (deleteError) {
        console.error(`Failed to delete chunk ${file.name}:`, deleteError);
      }
    }
  }

  // send message when done
  res.json({message: 'done'});
});

app.post('/api/summarize', async (req, res) => {
  const {videoId, token} = req.body;

  const supabase = await supabaseClient(token);

  const openai = new OpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: 'gpt-3.5-turbo-16k-0613',
    maxTokens: 500,
  });

  // try to download the summary file
  const {data: summaryDownloadData, error: summaryDownloadError} =
    await supabase.storage.from('summaries').download(`${videoId}/summary.txt`);

  console.log('summaryDownloadError', summaryDownloadError);

  if (summaryDownloadError) {
    if (summaryDownloadError.message === 'Object not found') {
      // if the summary file does not exist, generate a new summary

      // download the transcription file
      const {data: downloadData, error: downloadError} = await supabase.storage
        .from('transcriptions')
        .download(`${videoId}/transcription.txt`);

      if (downloadError) throw downloadError;

      // convert Blob to text
      const fullTranscription = await downloadData.text();

      // split the transcription into chunks of 10000 characters
      const textSplitter = new RecursiveCharacterTextSplitter({
        separators: ['\n\n', '\n'],
        chunkSize: 10000,
        chunkOverlap: 500,
      });
      const docs = await textSplitter.createDocuments([fullTranscription]);

      // define the summarization prompt
      const summarizationPromptTemplate = new PromptTemplate({
        template:
          'Read the following text and provide the key points to have a basic understanding as if I watched the video:\n"{content}"\nSUMMARY:',
        inputVariables: ['content'],
      });

      // create the LLMChain for summarization
      const summaryChain = new LLMChain({
        llm: openai,
        prompt: summarizationPromptTemplate,
      });

      console.log('Summarizing...');

      // run the summarization chain for each chunk and combine the summaries
      let combinedSummary = '';
      for (const doc of docs) {
        const summary = await summaryChain.run(doc.pageContent);
        combinedSummary += summary + '\n';
      }

      console.log(combinedSummary);

      // upload the combined summary to Supabase
      const {error: uploadError} = await supabase.storage
        .from('summaries')
        .upload(`${videoId}/summary.txt`, new Blob([combinedSummary]));

      if (uploadError) throw uploadError;
      console.log('Summary uploaded to Supabase', combinedSummary);

      // send message when done
      res.json({message: 'done', summary: combinedSummary});
    } else {
      // if there is another error, throw it
      console.log(summaryDownloadError);
    }
  } else {
    // if the summary file exists, return the summary
    try {
      const summary = await summaryDownloadData.text();
      console.log('Summary already exists', summary);
      res.status(200).json({message: 'Summary already exists', summary});
    } catch (error) {
      console.log(error);
    }
  }
});

app.listen(port, () => {
  console.log(`⚡️[server]: Server is running at http://localhost:${port}`);
});
