package parser

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
)

func Test_Parser(t *testing.T) {

	input := `
FROM model1
ADAPTER adapter1
LICENSE MIT
PARAMETER param1 value1
PARAMETER param2 value2
TEMPLATE template1
`

	reader := strings.NewReader(input)

	commands, err := Parse(reader)
	assert.Nil(t, err)

	expectedCommands := []Command{
		{Name: "model", Args: "model1"},
		{Name: "adapter", Args: "adapter1"},
		{Name: "license", Args: "MIT"},
		{Name: "param1", Args: "value1"},
		{Name: "param2", Args: "value2"},
		{Name: "template", Args: "template1"},
	}

	assert.True(t, cmp.Equal(expectedCommands, commands, cmpopts.IgnoreFields(Command{}, "Buffer")))
}

func Test_Parser_NoFromLine(t *testing.T) {

	input := `
PARAMETER param1 value1
PARAMETER param2 value2
`

	reader := strings.NewReader(input)

	_, err := Parse(reader)
	assert.ErrorContains(t, err, "no FROM line")
}

func Test_Parser_MissingValue(t *testing.T) {

	input := `
FROM foo
PARAMETER param1
`

	reader := strings.NewReader(input)

	_, err := Parse(reader)
	assert.ErrorContains(t, err, "missing value for [param1]")

}

func Test_Parser_Messages(t *testing.T) {

	input := `
FROM foo
MESSAGE system You are a Parser. Always Parse things.
MESSAGE user Hey there!
MESSAGE assistant Hello, I want to parse all the things!
`

	reader := strings.NewReader(input)
	commands, err := Parse(reader)
	assert.Nil(t, err)

	expectedCommands := []Command{
		{Name: "model", Args: "foo"},
		{Name: "message", Args: "system: You are a Parser. Always Parse things."},
		{Name: "message", Args: "user: Hey there!"},
		{Name: "message", Args: "assistant: Hello, I want to parse all the things!"},
	}

	assert.True(t, cmp.Equal(expectedCommands, commands, cmpopts.IgnoreFields(Command{}, "Buffer")))
}

func Test_Parser_Messages_BadRole(t *testing.T) {

	input := `
FROM foo
MESSAGE badguy I'm a bad guy!
`

	reader := strings.NewReader(input)
	_, err := Parse(reader)
	assert.ErrorContains(t, err, "role must be one of \"system\", \"user\", or \"assistant\"")
}

func Test_Parser_Multiline(t *testing.T) {
	type testCase struct {
		input    string
		expected []Command
	}

	var testCases = []testCase{
		{
			`
FROM foo
TEMPLATE """
{{ .System }}

{{ .Prompt }}
"""

SYSTEM """
This is a multiline system message.
"""
`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "{{ .System }}\n\n{{ .Prompt }}\n"},
				{Name: "system", Args: "This is a multiline system message.\n"},
			},
		},
		{
			`FROM foo
			TEMPLATE """{{ .System }} {{ .Prompt }}"""`,
			[]Command{
				{Name: "model", Args: "foo"},
				{Name: "template", Args: "{{ .System }} {{ .Prompt }}"},
			},
		},
	}

	for _, tc := range testCases {
		reader := strings.NewReader(tc.input)
		commands, err := Parse(reader)
		assert.Nil(t, err)

		assert.True(t, cmp.Equal(tc.expected, commands, cmpopts.IgnoreFields(Command{}, "Buffer")))
	}
}
