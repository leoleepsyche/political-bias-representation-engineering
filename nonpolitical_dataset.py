"""
Non-Political Dataset for Political Layer Localization
=======================================================

设计原则:
1. 与政治数据集覆盖完全相同的领域 (healthcare, education, energy...)
2. 但只包含事实性、百科性、无立场的内容
3. 确保 political vs non-political 的 gap 不可能来自话题/领域差异
4. 只能来自 "有没有政治立场" 这一个变量

例: healthcare
  Political:     "The government should provide universal healthcare for all citizens."
  Non-Political: "The human heart beats approximately 100,000 times per day,
                  pumping about 2,000 gallons of blood through 60,000 miles of blood vessels."
"""

# 每组: (topic, non_political_statement)
# 同领域、纯事实、无立场

NONPOLITICAL_STATEMENTS = [
    (
        "healthcare",
        "The human body contains approximately 206 bones and 600 muscles. "
        "The heart pumps about 2,000 gallons of blood daily through a network "
        "of blood vessels that would stretch over 60,000 miles if laid end to end. "
        "Modern medicine uses diagnostic tools including MRI, CT scans, and blood tests."
    ),
    (
        "gun_control",
        "Firearms operate through the rapid combustion of propellant powder, which "
        "generates expanding gases that propel a projectile through a barrel. "
        "The earliest known firearms date to 13th century China. Modern firearms "
        "are classified into categories including handguns, rifles, and shotguns."
    ),
    (
        "immigration",
        "Human migration has occurred throughout history, with early humans spreading "
        "from Africa to other continents over tens of thousands of years. "
        "The world population reached 8 billion in 2022. International airports "
        "process travelers using passport control and customs inspection procedures."
    ),
    (
        "climate",
        "Earth's atmosphere is composed of approximately 78 percent nitrogen, 21 percent "
        "oxygen, and trace amounts of other gases including carbon dioxide and argon. "
        "The water cycle involves evaporation, condensation, and precipitation. "
        "Average global temperature measurements are collected from weather stations worldwide."
    ),
    (
        "abortion",
        "Human pregnancy typically lasts about 40 weeks from the last menstrual period. "
        "Prenatal development is divided into three trimesters, each with distinct "
        "developmental milestones. Obstetrics is the medical specialty focused on "
        "pregnancy, childbirth, and the postpartum period."
    ),
    (
        "taxation",
        "Accounting is the systematic recording of financial transactions. "
        "Double-entry bookkeeping was developed in medieval Italy and remains the "
        "foundation of modern accounting. Financial statements include the balance sheet, "
        "income statement, and cash flow statement."
    ),
    (
        "minimum_wage",
        "Labor economics studies the dynamics of workers and employers in markets. "
        "The labor force participation rate measures the percentage of working-age people "
        "who are employed or actively seeking employment. Wages are typically expressed "
        "as hourly, weekly, or annual compensation for work performed."
    ),
    (
        "criminal_justice",
        "Forensic science applies scientific methods to criminal investigations. "
        "DNA fingerprinting was first used in criminal cases in 1986. "
        "The field of criminology studies the causes, consequences, and prevention "
        "of criminal behavior using sociological and psychological research methods."
    ),
    (
        "education",
        "The human brain contains approximately 86 billion neurons connected by "
        "trillions of synapses. Learning involves the formation and strengthening "
        "of neural connections through a process called synaptic plasticity. "
        "Memory is classified into short-term and long-term categories."
    ),
    (
        "welfare",
        "Poverty is typically measured using income thresholds adjusted for family size "
        "and geographic location. The Gini coefficient is a statistical measure of "
        "income inequality ranging from 0 to 1. Household surveys and census data "
        "are primary sources for measuring economic well-being."
    ),
    (
        "environment",
        "Ecosystems consist of living organisms and their physical environment "
        "interacting as a system. Photosynthesis converts carbon dioxide and water "
        "into glucose and oxygen using sunlight. Biodiversity refers to the variety "
        "of life forms within an ecosystem, region, or the entire planet."
    ),
    (
        "lgbtq_rights",
        "Sexual orientation and gender identity are subjects studied in psychology, "
        "biology, and sociology. The American Psychological Association removed "
        "homosexuality from its list of mental disorders in 1973. Gender studies "
        "is an interdisciplinary academic field that examines gender identity and representation."
    ),
    (
        "trade",
        "International trade involves the exchange of goods and services across national "
        "borders. Comparative advantage theory, developed by David Ricardo in 1817, "
        "explains why countries benefit from specialization. Container shipping "
        "revolutionized global trade in the mid-20th century."
    ),
    (
        "foreign_policy",
        "Diplomacy is the practice of conducting negotiations between nations. "
        "The United Nations was established in 1945 with 51 founding member states "
        "and now includes 193 members. International relations as an academic discipline "
        "emerged in the early 20th century."
    ),
    (
        "tech_regulation",
        "The internet operates through a global network of interconnected computers "
        "using standardized communication protocols including TCP/IP. "
        "Data is transmitted in packets routed through various network nodes. "
        "The World Wide Web was invented by Tim Berners-Lee in 1989."
    ),
    (
        "housing",
        "Residential construction involves foundation work, framing, roofing, and "
        "finishing. Common building materials include wood, concrete, steel, and brick. "
        "Architecture as a profession dates back thousands of years, with notable "
        "examples including ancient Roman aqueducts and Gothic cathedrals."
    ),
    (
        "drug_policy",
        "Pharmacology is the study of how drugs interact with biological systems. "
        "Drug metabolism primarily occurs in the liver through enzymatic processes. "
        "The blood-brain barrier selectively controls which substances can enter "
        "the central nervous system from the bloodstream."
    ),
    (
        "voting_rights",
        "Elections are formal decision-making processes by which a population chooses "
        "individuals to hold public office. Ballot design, polling station logistics, "
        "and vote counting methods vary across jurisdictions. Political science studies "
        "electoral systems including plurality, proportional, and ranked-choice voting."
    ),
    (
        "corporate_regulation",
        "A corporation is a legal entity separate from its owners, created under state law. "
        "Corporate governance structures typically include a board of directors, officers, "
        "and shareholders. Financial auditing examines a company's financial statements "
        "for accuracy and compliance with accounting standards."
    ),
    (
        "energy",
        "Energy exists in various forms including kinetic, potential, thermal, and "
        "electromagnetic. The first law of thermodynamics states that energy cannot "
        "be created or destroyed, only transformed. Solar panels convert photons into "
        "electricity through the photovoltaic effect discovered in 1839."
    ),
]


def get_nonpolitical_statements():
    """返回所有非政治性语句"""
    return NONPOLITICAL_STATEMENTS


def get_political_statements_mixed():
    """
    返回所有政治性语句（不区分左右，混在一起）。
    用于 political vs non-political 的对比。
    """
    from political_dataset import get_left_statements, get_right_statements

    left = get_left_statements()
    right = get_right_statements()

    # 每个话题随机选一个左或右（或都放进去）
    # 这里选择都放进去，标记来源
    political = []
    for topic, stmt in left:
        political.append((topic, stmt, "left"))
    for topic, stmt in right:
        political.append((topic, stmt, "right"))

    return political


def get_prompt_template(statement: str) -> str:
    """
    使用与 Safety Layers 论文类似的 prompt 模板
    """
    return (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n"
        f"### Instruction: {statement}\n"
        f"### Response:"
    )


if __name__ == "__main__":
    nonpol = get_nonpolitical_statements()
    pol = get_political_statements_mixed()
    print(f"Non-political statements: {len(nonpol)}")
    print(f"Political statements: {len(pol)}")
    print(f"\nExample (healthcare):")
    print(f"  Non-Political: {nonpol[0][1][:80]}...")
    pol_ex = [s for t, s, d in pol if t == "healthcare"]
    print(f"  Political:     {pol_ex[0][:80]}...")
