
import json
import os

notebook_path = r'c:\Users\Admin\OneDrive\Desktop\earthquake\analysis.ipynb'

def create_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }

def create_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    }

new_cells = [
    create_markdown_cell([
        "## 3.1 Data Search and Full View",
        "Use the cells below to view the full dataset and filter based on specific criteria."
    ]),
    create_code_cell([
        "# View Full Dataset (First 100 rows shown by default, modify n to see more)",
        "# Using option_context to temporarily show all columns and more rows",
        "with pd.option_context('display.max_rows', 100, 'display.max_columns', None):",
        "    display(df.head(100))"
    ]),
    create_code_cell([
        "# Search / Filter Data",
        "# Example: Find earthquakes with Magnitude > 7.0 and Depth < 50km",
        "search_criteria = (df['Magnitude'] > 7.0) & (df['Depth'] < 50)",
        "filtered_data = df[search_criteria]",
        "",
        "print(f'Found {len(filtered_data)} records matching criteria:')",
        "filtered_data.head(20)"
    ])
]

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the index of "Data Overview & Cleaning"
    insert_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            if "## 3. Data Overview & Cleaning" in source:
                # We want to insert AFTER section 3 cells. 
                # Let's look for the START of section 4 instead to insert BEFORE it.
                pass
            if "## 4. Exploratory Data Analysis (EDA)" in source:
                insert_idx = i
                break
    
    if insert_idx != -1:
        # Insert new cells before Section 4
        for cell in reversed(new_cells):
            nb['cells'].insert(insert_idx, cell)
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Notebook updated successfully.")
    else:
        print("Target cell (Section 4) not found. Checking for end of file.")
        nb['cells'].extend(new_cells)
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=4)
        print("Notebook updated (appended).")

except Exception as e:
    print(f"Error updating notebook: {e}")
