import { StartButton } from '@/components/StartButton';
import { ThemeToggle } from '@/components/ThemeToggle';

export const dynamic = 'force-dynamic';

export default async function Landing({
  searchParams,
}: {
  searchParams: Promise<{ src?: string }>;
}) {
  const { src } = await searchParams;

  return (
    <div className="page">
      <div className="row" style={{ marginBottom: 28 }}>
        <span className="eyebrow" style={{ margin: 0 }}>
          Tutor turn labelling
        </span>
        <span className="spacer" />
        <ThemeToggle />
      </div>

      <h1>Judge one thing about one tutor turn.</h1>
      <p className="lead">
        You will see a maths or science problem, what a student said, and one reply from an AI
        tutor. You answer a single question about that reply — usually yes or no — and move on. The
        anchors that define the question stay on screen the whole time.
      </p>

      <div className="card" style={{ marginTop: 28 }}>
        <div className="eyebrow">What it involves</div>
        <ul>
          <li>
            A short calibration set of about 15 items with the answers shown, so we both find out
            whether the question means the same thing to you as it does to us.
          </li>
          <li>
            Then sets of twelve turns. A set takes roughly five minutes. Stop whenever you like —
            every answer is saved the moment you give it.
          </li>
          <li>Keyboard throughout. Number keys answer; you never need the mouse.</li>
        </ul>
      </div>

      <div className="card">
        <div className="eyebrow">What we collect</div>
        <ul>
          <li>Your answers, and how long each one took.</li>
          <li>A country code, a shortened browser string, and a salted hash of your IP address.</li>
          <li>
            <strong>No name, no email, no account.</strong> You are identified only by a random
            handle in a cookie on this device.
          </li>
          <li>
            The labels will be published as a research dataset. Nothing in it can be traced to you.
          </li>
        </ul>
        <p style={{ fontSize: 14 }}>
          Starting means you are happy with the above. There is no way to withdraw a specific
          answer afterwards, because nothing links an answer to a person.
        </p>
      </div>

      <div className="row" style={{ marginTop: 28 }}>
        <StartButton src={src ?? null} />
      </div>
    </div>
  );
}
