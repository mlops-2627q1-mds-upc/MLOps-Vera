# Architecture diagrams (Milestone 4)

| Diagram | View | What it shows |
|---|---|---|
| [`logical-view.drawio`](logical-view.drawio) | Logical (UML component diagram) | The parts of the prediction service, where the model files come from, and the requirements each part covers |
| [`physical-view.drawio`](physical-view.drawio) | Physical (UML deployment diagram) | Where the service runs on the Virtech VM, and the protocol and port of every connection |
| [`predict-sequence.drawio`](predict-sequence.drawio) | Behaviour (UML sequence diagram) | One `POST /predict` call, including its error paths |

Each diagram has a PNG preview next to it. Vector PDFs for the report are in
[`reports/images/architecture/`](../../reports/images/architecture/):

```latex
\includegraphics[width=\textwidth]{images/architecture/logical-view.pdf}
```

The `FR-` and `NFR-` tags refer to the requirements in
[`2.1.1-requirements-engineering.md`](../../reports/milestone-1/2.1.1-requirements-engineering.md).
In the physical view, *n* is the last byte of the VM's private IP address (172.16.4.*n*).

## Editing

Open a `.drawio` file in the [draw.io desktop app](https://www.drawio.com/) or at
[app.diagrams.net](https://app.diagrams.net/). After an edit, export the PDF and the PNG again
from the repository root:

```bash
DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io  # macOS; on Linux: drawio
for f in docs/architecture/*.drawio; do
  name=$(basename "$f" .drawio)
  "$DRAWIO" --export --format pdf --crop --output "reports/images/architecture/$name.pdf" "$f"
  "$DRAWIO" --export --format png --scale 2 --border 12 --output "docs/architecture/$name.png" "$f"
done
```
