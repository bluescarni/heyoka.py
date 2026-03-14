{# Template for classes without a user-facing constructor (e.g., llvm_state). #}
{# Extends the base template but excludes __init__ from both :special-members: #}
{# and the methods autosummary table. The methods block below duplicates the   #}
{# logic from custom-class-base-template.rst with an added __init__ filter -  #}
{# keep it in sync if the base template's methods block changes.               #}
{% extends "custom-class-base-template.rst" %}
{% block methods %}
   {% if methods or '__call__' in members %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in members %}
      {% if (item in methods or item == '__call__') and item != '__init__' %}
         ~{{ name }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
{% endblock %}
