import Link from 'next/link';

export const dynamic = 'force-dynamic';

/**
 * One screen, not a document. Four rules, because these are the ones with measured effects on
 * agreement - checking against a shown reference solution alone took one question from 0.25 to
 * 0.75 - and a fifth would be read by nobody.
 */
export default function How() {
  return (
    <div className="page">
      <span className="eyebrow">Before you start</span>
      <h1>How to answer</h1>
      <p className="lead">Four rules. They are the whole method.</p>

      <div className="card" style={{ marginTop: 24 }}>
        <div className="eyebrow">1</div>
        <p className="lead">
          You are judging whether the turn <strong>does</strong> the thing, not whether it is good
          teaching.
        </p>
        <p>
          A clumsy turn that does the thing is a yes. An elegant turn that does not is a no. You are
          not scoring quality.
        </p>
      </div>

      <div className="card">
        <div className="eyebrow">2</div>
        <p className="lead">
          Answer by <strong>pointing at the text</strong>.
        </p>
        <p>
          When you say the turn does something, you will be asked which part — one keystroke. If you
          cannot point at a part, and you are working from an overall impression, that is what the
          flag key is for.
        </p>
      </div>

      <div className="card">
        <div className="eyebrow">3</div>
        <p className="lead">
          Check factual claims <strong>against the reference solution shown</strong>. Do not
          re-derive them.
        </p>
        <p>
          Where a question needs the worked solution, it is on screen. Making that change on one
          question in an earlier round moved agreement from 0.25 to 0.75.
        </p>
      </div>

      <div className="card">
        <div className="eyebrow">4</div>
        <p className="lead">
          You are <strong>not</strong> judging whether a different response would have been better.
        </p>
        <p>
          Three tutors pick the same next move in 18% of cases, so that question has no recoverable
          answer and we do not ask it.
        </p>
      </div>

      <div className="row" style={{ marginTop: 28 }}>
        <Link href="/calibration">
          <button className="primary">Start calibration</button>
        </Link>
        <span className="meta">about 15 items, answers shown as you go</span>
      </div>
    </div>
  );
}
