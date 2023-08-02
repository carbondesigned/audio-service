import express, {Express, Request, Response} from 'express';
import dotenv from 'dotenv';
import ytdl from 'ytdl-core';
import fs from 'fs';
import cp from 'child_process';

import {createClient} from '@supabase/supabase-js';

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

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

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
        for (let i = 0; i < 1000; i++) {
          // assuming a maximum of 1000 files
          const filePath = `${outputDir}/output${i
            .toString()
            .padStart(3, '0')}.mp3`;
          console.log(filePath);
          if (fs.existsSync(filePath)) {
            await uploadFile(filePath);
            fs.unlinkSync(filePath); // delete the local file after it's uploaded
          } else {
            break; // stop the loop if file doesn't exist
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

  console.log('idk');

  res.json({message: 'done'});
});

app.listen(port, () => {
  console.log(`⚡️[server]: Server is running at http://localhost:${port}`);
});